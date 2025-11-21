#include "yolov5_trt_bt.h"
#include "bytetrack.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unordered_map>
#include <deque>
#include <vector>
#include <cmath>
#include <numeric>
#include <chrono>

using namespace std;

// ======================= CONFIG =======================
static constexpr int   INPUT_W       = 640;
static constexpr int   INPUT_H       = 640;
static constexpr float CONF_HIGH     = 0.30f;  // conf fuerte
static constexpr float CONF_LOW      = 0.20f;  // conf débil
static constexpr float IOU_NMS_THR   = 0.65f;
static constexpr float IOU_MATCH_THR = 0.20f;
static constexpr int   MAX_TIME_LOST = 8;

// ======================= UTILES =======================
template<typename T>
static inline T clamp_(T v, T lo, T hi){ return v < lo ? lo : (v > hi ? hi : v); }

static float iou_rect(const cv::Rect2f& a, const cv::Rect2f& b){
    float inter = (a & b).area();
    float uni   = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.f;
}

static vector<int> nms(const vector<cv::Rect2f>& boxes, const vector<float>& scores, float iouTh){
    vector<int> order(boxes.size());
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(),
         [&](int a, int b){ return scores[a] > scores[b]; });
    vector<int> keep; vector<char> rem(boxes.size(), 0);
    for(size_t m=0; m<order.size(); ++m){
        int i = order[m]; if(rem[i]) continue; keep.push_back(i);
        for(size_t n=m+1; n<order.size(); ++n){
            int j = order[n]; if(rem[j]) continue;
            if(iou_rect(boxes[i], boxes[j]) > iouTh) rem[j] = 1;
        }
    }
    return keep;
}

// ======================= TRAIL =======================
struct Trail {
    deque<cv::Point> pts;
    static const int MAXLEN = 64;
    void add(const cv::Rect2f& bb){
        cv::Point c(bb.x + bb.width*0.5f, bb.y + bb.height*0.5f);
        pts.push_back(c);
        if(pts.size() > MAXLEN) pts.pop_front();
    }
};

// ======================= MAIN =======================
int main(int argc, char** argv){
    string engine_path = "/home/andfs/portal-inteligente-personas/modelos/YoloV5s/100_epocas_v2/best_yolov5s_100epochs.engine";
    string input_dir   = "/home/andfs/portal-inteligente-personas/test";
    string output_dir  = "/home/andfs/portal-inteligente-personas/resultyolov5bytetrack";

    system(("mkdir -p " + output_dir).c_str());
    YoLov5TRT_BT detector(engine_path);

    vector<string> videos;
    cv::glob(input_dir + "/*.mp4", videos, false);
    cout << "📂 Encontrados " << videos.size() << " videos en " << input_dir << endl;

    for(const auto& path : videos){
        cout << "▶ Procesando: " << path << endl;
        cv::VideoCapture cap(path);
        if(!cap.isOpened()){ cerr << "❌ No se pudo abrir: " << path << endl; continue; }

        int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps_in = cap.get(cv::CAP_PROP_FPS);
        if (fps_in <= 0) fps_in = 30;

        string base = path.substr(path.find_last_of("/\\") + 1);
        string out_path = output_dir + "/" + base;
        cv::VideoWriter writer(out_path,
            cv::VideoWriter::fourcc('a','v','c','1'),
            fps_in, cv::Size(W,H));

        // Tracking y contadores
        ByteTracker tracker(MAX_TIME_LOST, IOU_MATCH_THR);
        unordered_map<int, Trail> trails;
        int count_in = 0, count_out = 0;
        int mid_y = H / 2;

        while(true){
            cv::Mat frame;
            if(!cap.read(frame) || frame.empty()) break;

            auto t0 = chrono::high_resolution_clock::now();
            auto raw = detector.infer(frame);
            auto t1 = chrono::high_resolution_clock::now();
            float ms  = chrono::duration<float, milli>(t1 - t0).count();
            float fps = ms > 0 ? 1000.f / ms : 0.f;

            float r = detector.last_scale();
            int dl = detector.last_pad_left();
            int dt = detector.last_pad_top();

            vector<cv::Rect2f> boxes;
            vector<float> scores;
            for(auto& d : raw){
                float cx=d[0], cy=d[1], w=d[2], h=d[3], conf=d[4];
                if(conf < 0.01f) continue;
                float x1=(cx-w*0.5f - dl)/r;
                float y1=(cy-h*0.5f - dt)/r;
                float x2=(cx+w*0.5f - dl)/r;
                float y2=(cy+h*0.5f - dt)/r;
                x1=clamp_(x1,0.f,(float)W-1); y1=clamp_(y1,0.f,(float)H-1);
                x2=clamp_(x2,0.f,(float)W-1); y2=clamp_(y2,0.f,(float)H-1);
                boxes.emplace_back(x1,y1,x2-x1,y2-y1);
                scores.push_back(conf);
            }

            vector<int> keep;
            if(!boxes.empty()) keep = nms(boxes, scores, IOU_NMS_THR);

            vector<Det> det_high, det_low;
            for(int k : keep){
                Det d; d.tlwh = boxes[k]; d.score = scores[k];
                if(d.score >= CONF_HIGH) det_high.push_back(d);
                else if(d.score >= CONF_LOW) det_low.push_back(d);
            }

            auto tracks = tracker.update(det_high, det_low, W, H);

            // ==== Dibujar ====
            cv::Mat vis = frame.clone();
            cv::line(vis, {0,mid_y}, {W,mid_y}, {255,0,255}, 2);

            for(auto& t : tracks){
                if(t.state != TrackState::Tracked) continue;
                trails[t.id].add(t.tlwh);

                cv::Scalar color(t.id*37%255, t.id*17%255, t.id*91%255);
                cv::rectangle(vis, t.tlwh, color, 2);
                cv::putText(vis, "ID:"+to_string(t.id),
                            {(int)t.tlwh.x, max(0,(int)t.tlwh.y-5)},
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, {255,255,255}, 1);

                auto& pts = trails[t.id].pts;
                if(pts.size()>1){
                    for(size_t i=1;i<pts.size();++i)
                        cv::line(vis, pts[i-1], pts[i], color, 2);
                    int prev_y = pts[pts.size()-2].y;
                    int curr_y = pts.back().y;
                    if(prev_y < mid_y && curr_y >= mid_y) count_in++;
                    else if(prev_y > mid_y && curr_y <= mid_y) count_out++;
                }
            }

            char buf[128];
            snprintf(buf, sizeof(buf), "Total:%d  OUT:%d  IN:%d  FPS:%.1f",
                     count_in+count_out, count_out, count_in, fps);
            int baseline=0;
            cv::Size tsize=cv::getTextSize(buf,cv::FONT_HERSHEY_DUPLEX,0.7,1,&baseline);
            int bg_h = tsize.height + 10;
            cv::rectangle(vis,{0,H-bg_h},{W,H},{0,255,0},cv::FILLED);
            cv::putText(vis,buf,{10,H-5},cv::FONT_HERSHEY_DUPLEX,0.7,{0,0,0},1,cv::LINE_AA);

            writer.write(vis);
        }

        writer.release();
        cout << "💾 Guardado: " << out_path << endl;
    }

    cout << "✅ Todos los videos procesados. Resultados en: " << output_dir << endl;
    return 0;
}
