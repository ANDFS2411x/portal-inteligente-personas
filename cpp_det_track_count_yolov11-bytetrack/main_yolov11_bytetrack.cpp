#include "yolov11_trt.h"
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
static constexpr float HIGH_CONF_THR = 0.45f;  // detecciones fuertes
static constexpr float LOW_CONF_THR  = 0.20f;  // detecciones débiles (segunda asociación)
static constexpr float IOU_MATCH_THR = 0.25f;  // IoU asociación
static constexpr float IOU_NMS_THR   = 0.65f;  // NMS
static constexpr int   MAX_TRACK_AGE = 6;
static constexpr int   CAP_QUEUE_MAX = 5;
static constexpr int   VIZ_QUEUE_MAX = 1;
static const bool      USE_GSTREAMER = false;

// ======================= UTILES =======================
template<typename T> static inline T clamp_(T v,T lo,T hi){return v<lo?lo:(v>hi?hi:v);}
static float iou_rect(const cv::Rect2f& a,const cv::Rect2f& b){
    float inter=(a&b).area(), uni=a.area()+b.area()-inter;
    return uni>0?inter/uni:0.f;
}
static std::vector<int> nms(const std::vector<cv::Rect2f>& boxes,
                            const std::vector<float>& scores,float thr){
    std::vector<int> idx(boxes.size()); std::iota(idx.begin(),idx.end(),0);
    std::sort(idx.begin(),idx.end(),[&](int a,int b){return scores[a]>scores[b];});
    std::vector<int> keep; std::vector<char> rem(boxes.size(),0);
    for(size_t i=0;i<idx.size();++i){
        int m=idx[i]; if(rem[m])continue; keep.push_back(m);
        for(size_t j=i+1;j<idx.size();++j){
            int n=idx[j]; if(rem[n])continue;
            if(iou_rect(boxes[m],boxes[n])>thr) rem[n]=1;
        }
    } return keep;
}

// ======================= COLA =======================
template<typename T>
class BoundedQueue{
public: explicit BoundedQueue(size_t m):maxsize(m){}
    void push(const T&v){std::lock_guard<std::mutex>l(mtx);if(q.size()>=maxsize)q.pop();q.push(v);cv.notify_one();}
    bool pop(T&o,std::atomic<bool>&stop,int to=200){
        std::unique_lock<std::mutex>l(mtx);
        if(!cv.wait_for(l,std::chrono::milliseconds(to),[&]{return!q.empty()||stop.load();}))return false;
        if(q.empty())return false; o=std::move(q.front());q.pop();return true;
    }
private:std::queue<T>q;size_t maxsize;std::mutex mtx;std::condition_variable cv;
};

// ======================= TRACKER (ByteTrack-lite) =======================
struct Track{
    int id; cv::Rect2f box; float score; int age; int hits; bool active;
    Track(int _id,const cv::Rect2f&b,float s):id(_id),box(b),score(s),age(0),hits(1),active(true){}
};

static float center_dist(const cv::Rect2f&a,const cv::Rect2f&b){
    cv::Point2f ca(a.x+a.width*0.5f,a.y+a.height*0.5f);
    cv::Point2f cb(b.x+b.width*0.5f,b.y+b.height*0.5f);
    return std::hypot(ca.x-cb.x,ca.y-cb.y);
}

// asociación IoU simple
static std::vector<std::pair<int,int>> associate(const std::vector<cv::Rect2f>& dets,
                                                 const std::vector<cv::Rect2f>& trks,
                                                 float thr){
    std::vector<std::pair<int,int>> matches;
    std::vector<char> usedD(dets.size(),0),usedT(trks.size(),0);
    for(size_t i=0;i<dets.size();++i){
        float bestIou=thr; int bestT=-1;
        for(size_t j=0;j<trks.size();++j){
            if(usedT[j])continue;
            float iou=iou_rect(dets[i],trks[j]);
            if(iou>bestIou){bestIou=iou;bestT=j;}
        }
        if(bestT>=0){matches.push_back({(int)i,bestT});usedD[i]=1;usedT[bestT]=1;}
    }
    return matches;
}

// ======================= GLOBAL =======================
struct VizPacket{cv::Mat frame;};
std::atomic<bool>g_stop(false);
BoundedQueue<cv::Mat>q_frames(CAP_QUEUE_MAX);
BoundedQueue<VizPacket>q_viz(VIZ_QUEUE_MAX);

// ======================= CAPTURE =======================
void thread_capture(){
    cv::VideoCapture cap;
    if(USE_GSTREAMER){
        std::string pipe="v4l2src device=/dev/video0 ! image/jpeg,framerate=30/1,width=640,height=480 ! "
                         "jpegparse ! nvjpegdec ! nvvidconv ! video/x-raw,format=BGR,width=640,height=480 ! "
                         "appsink drop=true max-buffers=1";
        cap.open(pipe,cv::CAP_GSTREAMER);
    }else{
        cap.open(0,cv::CAP_V4L2);
        if(cap.isOpened()){
            cap.set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('M','J','P','G'));
            cap.set(cv::CAP_PROP_FRAME_WIDTH,640);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT,480);
        }
    }
    if(!cap.isOpened()){std::cerr<<"❌ No se pudo abrir la cámara\n";g_stop.store(true);return;}
    while(!g_stop.load()){cv::Mat f;if(!cap.read(f)||f.empty())break;q_frames.push(f);}
    g_stop.store(true);
}

// ======================= DETECCIÓN + BYTETRACK =======================
void thread_detect_track(const std::string&engine_path){
    YoLov11TRT detector(engine_path);
    std::vector<Track>active,lost;
    std::unordered_map<int,std::vector<cv::Point>>trail;
    std::vector<int>counted;
    int next_id=0,in_cnt=0,out_cnt=0;

    while(!g_stop.load()){
        cv::Mat frame; if(!q_frames.pop(frame,g_stop,200))continue;
        auto t0=std::chrono::high_resolution_clock::now();
        auto raw=detector.infer(frame);
        auto t1=std::chrono::high_resolution_clock::now();
        float ms=std::chrono::duration<float,std::milli>(t1-t0).count();
        float fps=ms>0?1000.f/ms:0.f;

        // deshacer letterbox
        float r=detector.last_scale();int dl=detector.last_pad_left();int dt=detector.last_pad_top();
        int W=frame.cols,H=frame.rows;

        // --- separar detecciones ---
        std::vector<cv::Rect2f> highB,lowB; std::vector<float> highS,lowS;
        for(auto&d:raw){
            float cx=d[0],cy=d[1],w=d[2],h=d[3],s=d[4];
            if(!std::isfinite(s)||s<LOW_CONF_THR)continue;
            float x1=(cx-w*0.5f-dl)/r, y1=(cy-h*0.5f-dt)/r;
            float x2=(cx+w*0.5f-dl)/r, y2=(cy+h*0.5f-dt)/r;
            x1=clamp_(x1,0.f,(float)W-1);y1=clamp_(y1,0.f,(float)H-1);
            x2=clamp_(x2,0.f,(float)W-1);y2=clamp_(y2,0.f,(float)H-1);
            if(x2<=x1||y2<=y1)continue;
            cv::Rect2f b(x1,y1,x2-x1,y2-y1);
            if(s>=HIGH_CONF_THR){highB.push_back(b);highS.push_back(s);}
            else {lowB.push_back(b);lowS.push_back(s);}
        }
        // NMS solo a high
        std::vector<int>keep;if(!highB.empty())keep=nms(highB,highS,IOU_NMS_THR);
        std::vector<cv::Rect2f>dets;std::vector<float>scores;
        for(int i:keep){dets.push_back(highB[i]);scores.push_back(highS[i]);}

        // --- predicciones actuales ---
        std::vector<cv::Rect2f>trk_boxes;
        for(auto&t:active)trk_boxes.push_back(t.box);

        // --- primera asociación (alta confianza) ---
        auto matches=associate(dets,trk_boxes,IOU_MATCH_THR);
        std::vector<int>usedD(dets.size(),0),usedT(active.size(),0);
        for(auto&m:matches){
            int di=m.first,ti=m.second;
            active[ti].box=dets[di];
            active[ti].score=scores[di];
            active[ti].age=0;active[ti].hits++;usedD[di]=1;usedT[ti]=1;
        }

        // --- trackers no emparejados ---
        std::vector<int>umT;
        for(size_t t=0;t<active.size();++t)if(!usedT[t])umT.push_back((int)t);

        // --- segunda asociación (detecciones bajas + lost) ---
        std::vector<cv::Rect2f>lost_boxes;
        for(auto&l:lost)lost_boxes.push_back(l.box);
        auto sec=associate(lowB,lost_boxes,IOU_MATCH_THR);
        std::vector<int>usedLow(lowB.size(),0),usedLost(lost.size(),0);
        for(auto&m:sec){
            int di=m.first,li=m.second;
            lost[li].box=lowB[di];
            lost[li].age=0;lost[li].hits++;lost[li].active=true;
            active.push_back(lost[li]);
            usedLow[di]=1;usedLost[li]=1;
        }

        // --- nuevos tracks ---
        for(size_t i=0;i<dets.size();++i)
            if(!usedD[i])active.emplace_back(next_id++,dets[i],scores[i]);

        // --- actualizar edad ---
        for(auto&t:active)t.age++;
        for(auto&t:lost)t.age++;

        // --- mover inactivos ---
        std::vector<Track>stillActive;stillActive.reserve(active.size());
        for(auto&t:active){
            if(t.age>MAX_TRACK_AGE){
                lost.push_back(t);
            }else stillActive.push_back(t);
        }
        active.swap(stillActive);

        // --- eliminar perdidos viejos ---
        lost.erase(std::remove_if(lost.begin(),lost.end(),
                   [](const Track&t){return t.age>MAX_TRACK_AGE*2;}),lost.end());

        // --- overlay ---
        cv::Mat vis=frame.clone();
        int yline=int(H*0.5f); cv::line(vis,{0,yline},{W,yline},{255,0,255},2);

        for(auto&t:active){
            cv::Scalar c(t.id*37%255,t.id*17%255,t.id*91%255);
            cv::rectangle(vis,t.box,c,2);
            cv::putText(vis,"ID:"+std::to_string(t.id),
                        {(int)t.box.x,std::max(0,(int)t.box.y-5)},
                        cv::FONT_HERSHEY_SIMPLEX,0.5,{255,255,255},1);
            cv::Point2f cp(t.box.x+t.box.width*0.5f,t.box.y+t.box.height*0.5f);
            trail[t.id].push_back(cp);
            auto&pts=trail[t.id];
            if(pts.size()>64)pts.erase(pts.begin(),pts.begin()+pts.size()-64);
            for(size_t i=1;i<pts.size();++i)cv::line(vis,pts[i-1],pts[i],c,2);

            if(!pts.empty()){
                int y0=pts.front().y,cy=cp.y;
                if(y0<H/2&&cy>H/2&&
                   std::find(counted.begin(),counted.end(),t.id)==counted.end()){
                    in_cnt++;counted.push_back(t.id);
                    std::cout<<"id:"<<t.id<<" OUT\n";
                }else if(y0>H/2&&cy<H/2&&
                         std::find(counted.begin(),counted.end(),t.id)==counted.end()){
                    out_cnt++;counted.push_back(t.id);
                    std::cout<<"id:"<<t.id<<" IN\n";
                }
            }
        }

        char buf[128];
        std::snprintf(buf,sizeof(buf),"Total:%d OUT:%d IN:%d FPS:%.1f",
                      in_cnt+out_cnt,in_cnt,out_cnt,fps);
        int base=0;cv::Size sz=cv::getTextSize(buf,cv::FONT_HERSHEY_DUPLEX,0.7,1,&base);
        cv::rectangle(vis,{0,H-sz.height-10},{W,H},{0,255,0},cv::FILLED);
        cv::putText(vis,buf,{10,H-5},cv::FONT_HERSHEY_DUPLEX,0.7,{0,0,0},1);
        q_viz.push(VizPacket{vis});
    }
    g_stop.store(true);
}

// ======================= VISUALIZAR =======================
void thread_viz(){
    cv::namedWindow("YOLOv11 ByteTrack",cv::WINDOW_NORMAL);
    cv::resizeWindow("YOLOv11 ByteTrack",800,450);
    while(!g_stop.load()){
        VizPacket pkt;if(!q_viz.pop(pkt,g_stop,500))continue;
        cv::imshow("YOLOv11 ByteTrack",pkt.frame);
        int k=cv::waitKey(1);
        if(k==27||k=='q'||k=='Q'){g_stop.store(true);break;}
    } cv::destroyAllWindows();
}

// ======================= MAIN =======================
int main(int argc,char**argv){
    std::string engine="/home/andfs/portal-inteligente-personas/modelos/YoloV11s/100_Épocas/best-yolov11-100.engine";
    if(argc>1)engine=argv[1];
    std::thread t1(thread_capture);
    std::thread t2(thread_detect_track,engine);
    std::thread t3(thread_viz);
    t1.join();t2.join();t3.join();
    return 0;
}
