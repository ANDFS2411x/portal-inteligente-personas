#include "yolov5_trt_bt.h"
#include "bytetrack.h"
#include <opencv2/opencv.hpp>
#include <atomic>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <cmath>
#include <numeric>

// ======================= CONFIG =======================
static constexpr int   INPUT_W       = 640;
static constexpr int   INPUT_H       = 640;
static constexpr float CONF_HIGH     = 0.30f;  // menos estricto
static constexpr float CONF_LOW      = 0.20f;  // detecciones débiles
static constexpr float IOU_NMS_THR   = 0.65f;
static constexpr float IOU_MATCH_THR = 0.20f;
static constexpr int   CAP_QUEUE_MAX = 5;
static constexpr int   VIZ_QUEUE_MAX = 1;
static constexpr int   MAX_TIME_LOST = 8;
static const bool      USE_GSTREAMER = false;

// ======================= ESTRUCTURAS =======================
struct FramePkt { cv::Mat frame; };
struct VizPkt   { cv::Mat annotated; };

std::atomic<bool> g_stop(false);

// ======================= BoundedQueue =======================
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
private:
    std::queue<T> q;
    size_t maxsize;
    std::mutex m;
    std::condition_variable cv_pop;
};

BoundedQueue<FramePkt> q_frames(CAP_QUEUE_MAX);
BoundedQueue<VizPkt>   q_viz(VIZ_QUEUE_MAX);

// ======================= UTILES =======================
template<typename T>
static inline T clamp_(T v, T lo, T hi){ return v < lo ? lo : (v > hi ? hi : v); }

static float iou_rect(const cv::Rect2f& a, const cv::Rect2f& b){
    float inter = (a & b).area();
    float uni   = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.f;
}

static std::vector<int> nms(const std::vector<cv::Rect2f>& boxes,
                            const std::vector<float>& scores, float iouTh){
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
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

// ======================= TRAIL =======================
struct Trail {
    std::deque<cv::Point> pts;
    static const int MAXLEN = 64;
    void add(const cv::Rect2f& bb){
        cv::Point c(bb.x + bb.width*0.5f, bb.y + bb.height*0.5f);
        pts.push_back(c);
        if(pts.size() > MAXLEN) pts.pop_front();
    }
};

// ======================= CAPTURA =======================
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
        FramePkt f;
        if (!cap.read(f.frame) || f.frame.empty()) break;
        q_frames.push(f);
    }
    g_stop.store(true);
}

// ======================= DETECCIÓN + TRACKING =======================
void thread_detect_track(const std::string& engine_path){
    YoLov5TRT_BT detector(engine_path);
    ByteTracker tracker(MAX_TIME_LOST, IOU_MATCH_THR);
    std::unordered_map<int, Trail> trails;

    int count_in = 0, count_out = 0;

    while(!g_stop.load()){
        FramePkt pkt;
        if (!q_frames.pop(pkt, g_stop, 200)) continue;
        cv::Mat frame = pkt.frame;
        int W = frame.cols, H = frame.rows;
        int mid_y = H / 2;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto raw = detector.infer(frame);
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms  = std::chrono::duration<float,std::milli>(t1 - t0).count();
        float fps = ms > 0 ? 1000.f/ms : 0.f;

        float r = detector.last_scale();
        int dl = detector.last_pad_left();
        int dt = detector.last_pad_top();

        std::vector<cv::Rect2f> boxes;
        std::vector<float> scores;
        for (auto& d : raw){
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

        std::vector<int> keep;
        if(!boxes.empty()) keep = nms(boxes, scores, IOU_NMS_THR);

        std::vector<Det> det_high, det_low;
        for (int k : keep){
            Det d; d.tlwh = boxes[k]; d.score = scores[k];
            if(d.score >= CONF_HIGH) det_high.push_back(d);
            else if(d.score >= CONF_LOW) det_low.push_back(d);
        }

        auto tracks = tracker.update(det_high, det_low, W, H);

        cv::Mat vis = frame.clone();
        cv::line(vis, {0,mid_y}, {W,mid_y}, {255,0,255}, 2);

        for(auto& t : tracks){
            if(t.state != TrackState::Tracked) continue;
            trails[t.id].add(t.tlwh);

            cv::Scalar color(t.id*37%255, t.id*17%255, t.id*91%255);
            cv::rectangle(vis, t.tlwh, color, 2);
            cv::putText(vis, "ID:"+std::to_string(t.id),
                        {(int)t.tlwh.x, std::max(0,(int)t.tlwh.y-5)},
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

        // HUD (Total / OUT / IN / FPS)
        char buf[128];
        std::snprintf(buf, sizeof(buf), "Total:%d  OUT:%d  IN:%d  FPS:%.1f",
                      count_in+count_out, count_in, count_out, fps);
        int baseline=0;
        cv::Size tsize=cv::getTextSize(buf, cv::FONT_HERSHEY_DUPLEX,0.7,1,&baseline);
        int bg_h = tsize.height + 10;
        cv::rectangle(vis,{0,H-bg_h},{W,H},{0,255,0},cv::FILLED);
        cv::putText(vis,buf,{10,H-5},cv::FONT_HERSHEY_DUPLEX,0.7,{0,0,0},1,cv::LINE_AA);

        q_viz.push(VizPkt{vis});
    }
    g_stop.store(true);
}

// ======================= VISUAL =======================
void thread_viz(){
    cv::namedWindow("Deteccion (preview)", cv::WINDOW_NORMAL);
    cv::resizeWindow("Deteccion (preview)", 800, 450);
    while(!g_stop.load()){
        VizPkt pkt;
        if(!q_viz.pop(pkt, g_stop, 500)) continue;
        cv::imshow("Deteccion (preview)", pkt.annotated);
        int k=cv::waitKey(1);
        if(k==27 || k=='q' || k=='Q'){ g_stop.store(true); break; }
    }
    cv::destroyAllWindows();
}

// ======================= MAIN =======================
int main(int argc, char** argv){
    std::string engine_path="/home/andfs/portal-inteligente-personas/modelos/YoloV5s/100_Epocas/best4_100epocas.engine";
    if(argc>1) engine_path=argv[1];
    std::thread t1(thread_capture);
    std::thread t2(thread_detect_track, engine_path);
    std::thread t3(thread_viz);
    t1.join(); t2.join(); t3.join();
    return 0;
}
