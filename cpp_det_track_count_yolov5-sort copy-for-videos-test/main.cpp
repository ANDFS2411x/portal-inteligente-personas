#include "yolov5_trt.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <numeric>

// ======================= CONFIG =======================
static constexpr int   INPUT_W       = 640;
static constexpr int   INPUT_H       = 640;
static constexpr float CONF_THRESH   = 0.30f;  // conf mínimo
static constexpr float IOU_NMS_THR   = 0.65f;  // NMS
static constexpr float IOU_MATCH_THR = 0.20f;  // asociación det<->track
static constexpr int   CAP_QUEUE_MAX = 5;      // cola captura → det
static constexpr int   VIZ_QUEUE_MAX = 1;      // cola det → viz
static constexpr int   MAX_TRACK_AGE = 6;      // frames sin update para borrar
static const bool      USE_GSTREAMER = false;  // fuente cámara

// ======================= Utiles =======================
template<typename T>
static inline T clamp_(T v, T lo, T hi){ return v < lo ? lo : (v > hi ? hi : v); }

static float iou_rect(const cv::Rect2f& a, const cv::Rect2f& b){
    float inter = (a & b).area();
    float uni   = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.f;
}

// NMS clásico (ordenado por score)
static std::vector<int> nms(const std::vector<cv::Rect2f>& boxes,
                            const std::vector<float>& scores, float iouTh){
    std::vector<int> order(boxes.size()); std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return scores[a] > scores[b]; });
    std::vector<int> keep; std::vector<char> rem(boxes.size(), 0);
    for(size_t m=0; m<order.size(); ++m){
        int i = order[m]; if(rem[i]) continue; keep.push_back(i);
        for(size_t n=m+1; n<order.size(); ++n){
            int j = order[n]; if(rem[j]) continue;
            if(iou_rect(boxes[i], boxes[j]) > iouTh) rem[j] = 1;
        }
    }
    return keep;
}

// ======================= BoundedQueue (drop oldest) =======================
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t maxsz): maxsize(maxsz) {}
    void push(const T& v){
        std::lock_guard<std::mutex> lk(m);
        if(q.size() >= maxsize) q.pop();
        q.push(v);
        cv_pop.notify_one();
    }
    bool pop(T& out, std::atomic<bool>& stop, int timeout_ms=200){
        std::unique_lock<std::mutex> lk(m);
        if(!cv_pop.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                            [&]{ return !q.empty() || stop.load(); }))
            return false;
        if(q.empty()) return false;
        out = std::move(q.front()); q.pop();
        return true;
    }
    void clear(){ std::lock_guard<std::mutex> lk(m); while(!q.empty()) q.pop(); }
private:
    std::queue<T> q;
    size_t maxsize;
    std::mutex m;
    std::condition_variable cv_pop;
};

// ======================= Kalman + conversiones =======================
// Estado: [x, y, s, r, vx, vy, vs]  (x,y centro; s=area; r=ratio w/h)
struct KBTracker {
    cv::KalmanFilter kf;
    int id = -1;
    int age = 0;
    int time_since_update = 0;
    int hits = 0;
    int hit_streak = 0;

    KBTracker(int _id, const cv::Rect2f& bb){
        id = _id;
        kf = cv::KalmanFilter(7,4,0, CV_32F);
        // F
        kf.transitionMatrix = (cv::Mat_<float>(7,7) <<
            1,0,0,0,1,0,0,
            0,1,0,0,0,1,0,
            0,0,1,0,0,0,1,
            0,0,0,1,0,0,0,
            0,0,0,0,1,0,0,
            0,0,0,0,0,1,0,
            0,0,0,0,0,0,1);

        // H
        kf.measurementMatrix = cv::Mat::zeros(4,7,CV_32F);
        for(int i=0;i<4;++i) kf.measurementMatrix.at<float>(i,i) = 1.f;

        // P (alta incertidumbre en velocidades)
        kf.errorCovPost = cv::Mat::eye(7,7,CV_32F);
        for(int r=4;r<7;++r) kf.errorCovPost.at<float>(r,r) *= 1000.f;
        kf.errorCovPost *= 10.f;

        // Q/R (puedes afinar)
        setIdentity(kf.processNoiseCov,     cv::Scalar(1e-2));
        setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));

        // estado inicial
        float w = bb.width, h = bb.height;
        float x = bb.x + w*0.5f, y = bb.y + h*0.5f;
        float s = w*h, r = w / std::max(h, 1e-6f);

        kf.statePost = cv::Mat::zeros(7,1,CV_32F);
        kf.statePost.at<float>(0)=x;
        kf.statePost.at<float>(1)=y;
        kf.statePost.at<float>(2)=s;
        kf.statePost.at<float>(3)=r;
    }

    cv::Rect2f predict(){
        if (kf.statePost.at<float>(6) + kf.statePost.at<float>(2) <= 0.f)
            kf.statePost.at<float>(6) = 0.f;
        cv::Mat x = kf.predict();
        age++;
        if(time_since_update > 0) hit_streak = 0;
        time_since_update++;

        float sx = x.at<float>(0);
        float sy = x.at<float>(1);
        float ss = x.at<float>(2);
        float rr = x.at<float>(3);
        float w = std::sqrt(std::max(0.f, ss*rr));
        float h = (w>0) ? ss / w : 0.f;
        float x1 = sx - w*0.5f, y1 = sy - h*0.5f;
        return cv::Rect2f(x1,y1,w,h);
    }

    void update(const cv::Rect2f& bb){
        time_since_update = 0;
        hits++; hit_streak++;
        cv::Mat z(4,1,CV_32F);
        float w = bb.width, h = bb.height;
        float x = bb.x + w*0.5f, y = bb.y + h*0.5f;
        float s = w*h, r = w / std::max(h, 1e-6f);
        z.at<float>(0)=x; z.at<float>(1)=y; z.at<float>(2)=s; z.at<float>(3)=r;
        kf.correct(z);
    }

    cv::Rect2f get_state() const {
        const cv::Mat& x = kf.statePost;
        float sx = x.at<float>(0);
        float sy = x.at<float>(1);
        float ss = x.at<float>(2);
        float rr = x.at<float>(3);
        float w = std::sqrt(std::max(0.f, ss*rr));
        float h = (w>0) ? ss / w : 0.f;
        float x1 = sx - w*0.5f, y1 = sy - h*0.5f;
        return cv::Rect2f(x1,y1,w,h);
    }
};

// ======================= Hungarian (mínimo coste) =======================
static void padSquare(std::vector<std::vector<float>>& m, float padVal){
    size_t n = m.size(), p = 0;
    for (auto& r : m) p = std::max(p, r.size());
    size_t N = std::max(n, p);
    m.resize(N, std::vector<float>(p, padVal));
    for (auto& r : m) r.resize(N, padVal);
}

// Implementación compacta estilo Munkres O(N^3)
static std::vector<int> hungarian(const std::vector<std::vector<float>>& cost){
    const int N = (int)cost.size();
    std::vector<float> u(N+1), v(N+1);
    std::vector<int> p(N+1), way(N+1);
    for (int i=1; i<=N; ++i){
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(N+1, 1e9f);
        std::vector<char> used(N+1, false);
        do{
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            float delta = 1e9f;
            for (int j=1; j<=N; ++j){
                if(used[j]) continue;
                float cur = cost[i0-1][j-1]-u[i0]-v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta){ delta = minv[j]; j1 = j; }
            }
            for (int j=0; j<=N; ++j){
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else          { minv[j]  -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do{
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    std::vector<int> ans(N, -1);
    for (int j=1; j<=N; ++j) if (p[j]>0) ans[p[j]-1] = j-1;
    return ans;
}

// Asociación detecciones<->trackers (coste = 1 - IoU)
static void associate_dets_tracks(const std::vector<cv::Rect2f>& dets,
                                  const std::vector<cv::Rect2f>& trks,
                                  float iouThr,
                                  std::vector<std::pair<int,int>>& matches,
                                  std::vector<int>& unmatched_dets,
                                  std::vector<int>& unmatched_trks)
{
    matches.clear(); unmatched_dets.clear(); unmatched_trks.clear();
    if (trks.empty()){
        unmatched_dets.resize(dets.size());
        std::iota(unmatched_dets.begin(), unmatched_dets.end(), 0);
        return;
    }
    std::vector<std::vector<float>> cost(dets.size(), std::vector<float>(trks.size(), 1.f));
    for (size_t d=0; d<dets.size(); ++d)
        for (size_t t=0; t<trks.size(); ++t)
            cost[d][t] = 1.f - iou_rect(dets[d], trks[t]);

    size_t origR=dets.size(), origC=trks.size();
    padSquare(cost, 1.f);
    std::vector<int> assign = hungarian(cost); // tamaño NxN, -1 si fila sin asignar

    std::vector<char> trk_used(origC, 0), det_used(origR, 0);
    for (size_t di=0; di<origR; ++di){
        int tj = assign[di];
        if (tj >= 0 && (size_t)tj < origC){
            float iou = 1.f - cost[di][tj];
            if (iou >= iouThr){
                matches.emplace_back((int)di, tj);
                trk_used[tj] = 1; det_used[di] = 1;
            }
        }
    }
    for (size_t d=0; d<origR; ++d) if (!det_used[d]) unmatched_dets.push_back((int)d);
    for (size_t t=0; t<origC; ++t) if (!trk_used[t]) unmatched_trks.push_back((int)t);
}

// ======================= Paquetes entre hilos =======================
struct VizPacket {
    cv::Mat frame_annot;  // frame ya dibujado (detecciones + tracks + texto)
};

// ======================= Globales =======================
std::atomic<bool> g_stop(false);
BoundedQueue<cv::Mat> q_frames(CAP_QUEUE_MAX);
BoundedQueue<VizPacket> q_viz(VIZ_QUEUE_MAX);

// ======================= thread_capture =======================
void thread_capture(){
    cv::VideoCapture cap;
    if (USE_GSTREAMER){
        std::string pipe =
            "v4l2src device=/dev/video0 ! image/jpeg,framerate=30/1,width=640,height=480 ! "
            "jpegparse ! nvjpegdec ! nvvidconv ! video/x-raw,format=BGR,width=640,height=480 ! "
            "appsink drop=true max-buffers=1";
        cap.open(pipe, cv::CAP_GSTREAMER);
    } else {
        cap.open(0, cv::CAP_V4L2);
        if (cap.isOpened()){
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
            cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        }
    }
    if (!cap.isOpened()){
        std::cerr << "❌ No se pudo abrir la cámara\n";
        g_stop.store(true);
        return;
    }
    while(!g_stop.load()){
        cv::Mat f;
        if (!cap.read(f) || f.empty()) break;
        q_frames.push(f);
    }
    g_stop.store(true);
}

// ======================= thread_detect_track =======================
void thread_detect_track(const std::string& engine_path){
    YoLov5TRT detector(engine_path);

    std::vector<KBTracker> trackers;
    std::unordered_map<int, std::vector<cv::Point>> trail;
    std::vector<int> id_counted; id_counted.reserve(1024);
    int next_id = 0;
    int in_cnt=0, out_cnt=0;

    while(!g_stop.load()){
        cv::Mat frame;
        if (!q_frames.pop(frame, g_stop, 200)) continue;

        auto t0 = std::chrono::high_resolution_clock::now();
        // === Inferencia TRT ===
        auto raw = detector.infer(frame); // N x 6
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms  = std::chrono::duration<float,std::milli>(t1 - t0).count();
        float fps = ms > 0 ? 1000.f/ms : 0.f;

        // === Deshacer letterbox ===
        float r  = detector.last_scale();
        int   dl = detector.last_pad_left();
        int   dt = detector.last_pad_top();

        int W = frame.cols, H = frame.rows;
        std::vector<cv::Rect2f> boxes; boxes.reserve(raw.size());
        std::vector<float> scores;     scores.reserve(raw.size());

        for(const auto& det: raw){
            float cx=det[0], cy=det[1], w=det[2], h=det[3], conf=det[4];
            if (!std::isfinite(conf) || conf < CONF_THRESH) continue;
            float x1l=cx-w*0.5f, y1l=cy-h*0.5f;
            float x2l=cx+w*0.5f, y2l=cy+h*0.5f;
            float x1=(x1l - dl)/r, y1=(y1l - dt)/r;
            float x2=(x2l - dl)/r, y2=(y2l - dt)/r;
            x1 = clamp_(x1, 0.f, (float)W-1); y1 = clamp_(y1, 0.f, (float)H-1);
            x2 = clamp_(x2, 0.f, (float)W-1); y2 = clamp_(y2, 0.f, (float)H-1);
            if (x2<=x1 || y2<=y1) continue;
            boxes.emplace_back(x1, y1, x2-x1, y2-y1);
            scores.push_back(conf);
        }

        // === NMS ===
        std::vector<int> keep;
        if (!boxes.empty()) keep = nms(boxes, scores, IOU_NMS_THR);

        std::vector<cv::Rect2f> dets; dets.reserve(keep.size());
        for(int k : keep) dets.push_back(boxes[k]);

        // === PREDICCIÓN de todos los trackers ===
        std::vector<cv::Rect2f> trk_preds;
        trk_preds.reserve(trackers.size());
        std::vector<int> to_del;
        for(size_t t=0; t<trackers.size(); ++t){
            cv::Rect2f bb = trackers[t].predict();
            if (!std::isfinite(bb.x) || !std::isfinite(bb.y) ||
                !std::isfinite(bb.width) || !std::isfinite(bb.height))
                to_del.push_back((int)t);
            else trk_preds.push_back(bb);
        }
        for (int i=(int)to_del.size()-1; i>=0; --i){
            int idx = to_del[i];
            if (idx>=0 && idx<(int)trackers.size()) trackers.erase(trackers.begin()+idx);
        }

        // === ASOCIACIÓN ===
        std::vector<std::pair<int,int>> matches;
        std::vector<int> um_dets, um_trks;
        associate_dets_tracks(dets, trk_preds, IOU_MATCH_THR, matches, um_dets, um_trks);

        // === ACTUALIZAR trackers existentes ===
        for (const auto& m : matches){
            int d = m.first, t = m.second;
            trackers[t].update(dets[d]);

            // rastro + conteo por cruce de línea media
            cv::Rect2f b = dets[d];
            int cx = int(b.x + b.width*0.5f);
            int cy = int(b.y + b.height*0.5f);
            trail[trackers[t].id].push_back({cx,cy});

            const auto& pts = trail[trackers[t].id];
            if (!pts.empty()){
                int y0 = pts.front().y;
                if (y0 < H/2 && cy > H/2 &&
                    std::find(id_counted.begin(), id_counted.end(), trackers[t].id) == id_counted.end()){
                    in_cnt++; id_counted.push_back(trackers[t].id);
                    std::cout << "id: " << trackers[t].id << " - OUT\n";
                } else if (y0 > H/2 && cy < H/2 &&
                           std::find(id_counted.begin(), id_counted.end(), trackers[t].id) == id_counted.end()){
                    out_cnt++; id_counted.push_back(trackers[t].id);
                    std::cout << "id: " << trackers[t].id << " - IN\n";
                }
            }
        }

        // === NUEVOS trackers por detecciones no emparejadas ===
        for (int d : um_dets){
            KBTracker trk(next_id++, dets[d]);
            trackers.push_back(std::move(trk));
            cv::Rect2f b = dets[d];
            int cx = int(b.x + b.width*0.5f);
            int cy = int(b.y + b.height*0.5f);
            trail[trackers.back().id].push_back({cx,cy});
        }

        // === ELIMINAR trackers viejos ===
        for (int i=(int)trackers.size()-1; i>=0; --i){
            if (trackers[i].time_since_update > MAX_TRACK_AGE){
                trackers.erase(trackers.begin()+i);
            }
        }

        // === DIBUJAR overlay ===
        cv::Mat vis = frame.clone();
        int yline = int(H * 0.5f);
        cv::line(vis, {0,yline}, {W,yline}, {255,0,255}, 2);

        for (auto& trk : trackers){
            cv::Rect2f b = trk.get_state();
            cv::Scalar color(trk.id*37%255, trk.id*17%255, trk.id*91%255);
            cv::rectangle(vis, b, color, 2);
            cv::putText(vis, "ID:"+std::to_string(trk.id),
                        { (int)b.x, std::max(0, (int)b.y-5) },
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, {255,255,255}, 1);

            auto it = trail.find(trk.id);
            if (it != trail.end() && it->second.size() > 1){
                const auto& pts = it->second;
                for (size_t i=1; i<pts.size(); ++i)
                    cv::line(vis, pts[i-1], pts[i], color, 2);
                if (it->second.size() > 64) {
                    it->second.erase(it->second.begin(), it->second.begin() + (it->second.size()-64));
                }
            }
        }

        char buf[128];
        std::snprintf(buf, sizeof(buf), "Total:%d  OUT:%d  IN:%d  FPS:%.1f",
                      in_cnt+out_cnt, in_cnt, out_cnt, fps);
        int baseline=0;
        cv::Size tsize = cv::getTextSize(buf, cv::FONT_HERSHEY_DUPLEX, 0.7, 1, &baseline);
        int bg_h = tsize.height + 10;
        cv::rectangle(vis, {0, H-bg_h}, {W,H}, {0,255,0}, cv::FILLED);
        cv::putText(vis, buf, {10, H-5}, cv::FONT_HERSHEY_DUPLEX, 0.7, {0,0,0}, 1, cv::LINE_AA);

        // === Enviar a visualización (drop-oldest) ===
        q_viz.push(VizPacket{vis});
    }
    g_stop.store(true);
}

// ======================= thread_viz =======================
void thread_viz(){
    cv::namedWindow("Deteccion (preview)", cv::WINDOW_NORMAL);
    cv::resizeWindow("Deteccion (preview)", 800, 450);

    while(!g_stop.load()){
        VizPacket pkt;
        if (!q_viz.pop(pkt, g_stop, 500)) continue;

        cv::imshow("Deteccion (preview)", pkt.frame_annot);
        int k = cv::waitKey(1);
        if (k==27 || k=='q' || k=='Q'){ g_stop.store(true); break; }
    }
    cv::destroyAllWindows();
}

// ======================= main =======================
int main(int argc, char** argv){
    std::string engine_path = "/home/andfs/portal-inteligente-personas/modelos/YoloV5s/100_epocas_v2/best_yolov5s_100epochs.engine";
    std::string input_dir   = "/home/andfs/portal-inteligente-personas/test";
    std::string output_dir  = "/home/andfs/portal-inteligente-personas/result";

    // Crea carpeta result si no existe
    system(("mkdir -p " + output_dir).c_str());

    // Inicializa detector TRT (una sola vez)
    YoLov5TRT detector(engine_path);

    // Itera todos los .mp4 del directorio
    std::vector<std::string> videos;
    cv::glob(input_dir + "/*.mp4", videos, false);
    std::cout << "📂 Encontrados " << videos.size() << " videos en " << input_dir << "\n";

    for (const auto& path : videos){
        std::cout << "▶ Procesando: " << path << std::endl;

        cv::VideoCapture cap(path);
        if (!cap.isOpened()){ std::cerr << "❌ No se pudo abrir: " << path << std::endl; continue; }

        int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps_in = cap.get(cv::CAP_PROP_FPS);
        if (fps_in <= 0) fps_in = 30;

        std::string base = path.substr(path.find_last_of("/\\") + 1);
        std::string out_path = output_dir + "/" + base;

        cv::VideoWriter writer(out_path,
            cv::VideoWriter::fourcc('a','v','c','1'),
            fps_in, cv::Size(W,H));

        // === Variables de tracking ===
        std::vector<KBTracker> trackers;
        std::unordered_map<int, std::vector<cv::Point>> trail;
        std::vector<int> id_counted;
        int next_id = 0;
        int in_cnt = 0, out_cnt = 0;

        while (true){
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) break;

            // ======= Inferencia =======
            auto raw = detector.infer(frame);
            float r  = detector.last_scale();
            int dl   = detector.last_pad_left();
            int dt   = detector.last_pad_top();

            std::vector<cv::Rect2f> boxes;
            std::vector<float> scores;
            for (const auto& det : raw){
                float cx=det[0], cy=det[1], w=det[2], h=det[3], conf=det[4];
                if (!std::isfinite(conf) || conf < CONF_THRESH) continue;
                float x1l=cx-w*0.5f, y1l=cy-h*0.5f;
                float x2l=cx+w*0.5f, y2l=cy+h*0.5f;
                float x1=(x1l - dl)/r, y1=(y1l - dt)/r;
                float x2=(x2l - dl)/r, y2=(y2l - dt)/r;
                x1 = clamp_(x1, 0.f, (float)W-1); y1 = clamp_(y1, 0.f, (float)H-1);
                x2 = clamp_(x2, 0.f, (float)W-1); y2 = clamp_(y2, 0.f, (float)H-1);
                if (x2<=x1 || y2<=y1) continue;
                boxes.emplace_back(x1, y1, x2-x1, y2-y1);
                scores.push_back(conf);
            }

            std::vector<int> keep;
            if (!boxes.empty()) keep = nms(boxes, scores, IOU_NMS_THR);
            std::vector<cv::Rect2f> dets;
            for(int k : keep) dets.push_back(boxes[k]);

            // ======= PREDICCIÓN =======
            std::vector<cv::Rect2f> trk_preds;
            for (auto& t : trackers) trk_preds.push_back(t.predict());

            // ======= ASOCIACIÓN =======
            std::vector<std::pair<int,int>> matches;
            std::vector<int> um_dets, um_trks;
            associate_dets_tracks(dets, trk_preds, IOU_MATCH_THR, matches, um_dets, um_trks);

            for (auto& m : matches){
                int d = m.first;
                int t = m.second;
                trackers[t].update(dets[d]);
                cv::Rect2f b = dets[d];
                int cx = int(b.x + b.width*0.5f);
                int cy = int(b.y + b.height*0.5f);
                trail[trackers[t].id].push_back({cx,cy});

                const auto& pts = trail[trackers[t].id];
                if (!pts.empty()){
                    int y0 = pts.front().y;
                    if (y0 < H/2 && cy > H/2 &&
                        std::find(id_counted.begin(), id_counted.end(), trackers[t].id) == id_counted.end()){
                        in_cnt++; id_counted.push_back(trackers[t].id);
                    } else if (y0 > H/2 && cy < H/2 &&
                               std::find(id_counted.begin(), id_counted.end(), trackers[t].id) == id_counted.end()){
                        out_cnt++; id_counted.push_back(trackers[t].id);
                    }
                }
            }

            for (int d : um_dets){
                KBTracker trk(next_id++, dets[d]);
                trackers.push_back(std::move(trk));
                cv::Rect2f b = dets[d];
                int cx = int(b.x + b.width*0.5f);
                int cy = int(b.y + b.height*0.5f);
                trail[trackers.back().id].push_back({cx,cy});
            }

            for (int i=(int)trackers.size()-1; i>=0; --i){
                if (trackers[i].time_since_update > MAX_TRACK_AGE)
                    trackers.erase(trackers.begin()+i);
            }

            // ======= VISUALIZACIÓN =======
            cv::Mat vis = frame.clone();
            int yline = int(H * 0.5f);
            cv::line(vis, {0,yline}, {W,yline}, {255,0,255}, 2);

            for (auto& trk : trackers){
                cv::Rect2f b = trk.get_state();
                cv::Scalar color(trk.id*37%255, trk.id*17%255, trk.id*91%255);
                cv::rectangle(vis, b, color, 2);
                cv::putText(vis, "ID:"+std::to_string(trk.id),
                            {(int)b.x, std::max(0,(int)b.y-5)},
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, {255,255,255}, 1);

                auto it = trail.find(trk.id);
                if (it != trail.end() && it->second.size() > 1){
                    const auto& pts = it->second;
                    for (size_t i=1;i<pts.size();++i)
                        cv::line(vis, pts[i-1], pts[i], color, 2);
                    if (it->second.size() > 64)
                        it->second.erase(it->second.begin(), it->second.begin()+(it->second.size()-64));
                }
            }

            char buf[128];
            std::snprintf(buf, sizeof(buf), "Total:%d  IN:%d  OUT:%d",
                          in_cnt+out_cnt, in_cnt, out_cnt);
            cv::rectangle(vis, {0,H-30},{W,H},{0,255,0},cv::FILLED);
            cv::putText(vis, buf, {10,H-10}, cv::FONT_HERSHEY_DUPLEX, 0.7, {0,0,0}, 1, cv::LINE_AA);

            writer.write(vis);
        }

        writer.release();
        std::cout << "💾 Guardado: " << out_path << std::endl;
    }

    std::cout << "✅ Todos los videos procesados. Resultados en: " << output_dir << std::endl;
    return 0;
}

