#ifndef BYTETRACK_H
#define BYTETRACK_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

// ======================= Estados =======================
enum class TrackState { Tracked, Lost, Removed };

struct Det {
    cv::Rect2f tlwh;   // x,y,w,h
    float score = 0.f; // confianza
};

static inline float iou_rect_bt(const cv::Rect2f& a, const cv::Rect2f& b){
    float inter = (a & b).area();
    float uni   = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.f;
}

// ======================= Kalman (lite) =======================
// Estado 7D (x,y,s,r, vx,vy,vs) y medición 4D (x,y,s,r) como ByteTrack.
// Ruido reducido para menos cómputo y más estabilidad en Nano.
struct BTKalman {
    cv::KalmanFilter kf;
    BTKalman(){
        kf = cv::KalmanFilter(7, 4, 0, CV_32F);
        kf.transitionMatrix = (cv::Mat_<float>(7,7) <<
            1,0,0,0,1,0,0,
            0,1,0,0,0,1,0,
            0,0,1,0,0,0,1,
            0,0,0,1,0,0,0,
            0,0,0,0,1,0,0,
            0,0,0,0,0,1,0,
            0,0,0,0,0,0,1);
        kf.measurementMatrix = cv::Mat::zeros(4,7,CV_32F);
        for (int i=0;i<4;++i) kf.measurementMatrix.at<float>(i,i) = 1.f;
        setIdentity(kf.processNoiseCov,     cv::Scalar(1e-3));  // lite
        setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
        kf.errorCovPost = cv::Mat::eye(7,7,CV_32F) * 5.f;       // menos grande
        kf.statePost    = cv::Mat::zeros(7,1,CV_32F);
    }
    static void tlwh_to_xyah(const cv::Rect2f& bb, cv::Mat& z){
        float w=bb.width, h=bb.height;
        float x=bb.x + w*0.5f, y=bb.y + h*0.5f;
        float s=w*h; float r=w/std::max(h, 1e-6f);
        z.at<float>(0)=x; z.at<float>(1)=y; z.at<float>(2)=s; z.at<float>(3)=r;
    }
    static cv::Rect2f xyah_to_tlwh(const cv::Mat& x){
        float cx=x.at<float>(0), cy=x.at<float>(1), s=x.at<float>(2), r=x.at<float>(3);
        float w=std::sqrt(std::max(0.f, s*r));
        float h=(w>0.f)? s/w : 0.f;
        return {cx-w*0.5f, cy-h*0.5f, w, h};
    }
    void init(const cv::Rect2f& bb){
        cv::Mat z(4,1,CV_32F); tlwh_to_xyah(bb,z);
        kf.statePost.setTo(0);
        for (int i=0;i<4;++i) kf.statePost.at<float>(i) = z.at<float>(i);
    }
    cv::Rect2f predict(){ cv::Mat x=kf.predict(); return xyah_to_tlwh(x); }
    cv::Rect2f update (const cv::Rect2f& bb){
        cv::Mat z(4,1,CV_32F); tlwh_to_xyah(bb,z);
        kf.correct(z); return xyah_to_tlwh(kf.statePost);
    }
};

// ======================= Track =======================
struct Track {
    int id=-1;
    TrackState state=TrackState::Tracked;
    int frame_age=0;          // frames totales
    int time_since_update=0;  // frames sin match
    float score=0.f;
    cv::Rect2f tlwh;
    BTKalman kf;

    Track()=default;
    explicit Track(int _id, const Det& d){
        id=_id; tlwh=d.tlwh; score=d.score;
        state=TrackState::Tracked; frame_age=1; time_since_update=0;
        kf.init(tlwh);
    }
    cv::Rect2f predict(){ tlwh=kf.predict(); frame_age++; time_since_update++; return tlwh; }
    void update (const Det& d){ tlwh=kf.update(d.tlwh); score=d.score; time_since_update=0; state=TrackState::Tracked; }
    void mark_lost(){ state=TrackState::Lost; }
    void mark_removed(){ state=TrackState::Removed; }
};

// ======================= ByteTrack LITE =======================
class ByteTracker {
public:
    // iou_match_thr: umbral mínimo de IoU para considerar match
    ByteTracker(int max_time_lost, float iou_match_thr)
        : max_time_lost_(max_time_lost), iou_thr_(iou_match_thr) {}

    // Entrada: detecciones high/low tras NMS
    // Salida: tracks activos (Tracked + Lost para diagnóstico)
    std::vector<Track> update(const std::vector<Det>& dets_high,
                              const std::vector<Det>& dets_low,
                              int /*img_w*/, int /*img_h*/)
    {
        // 1) predecir todos
        for (auto& t : tracked_) t.predict();
        for (auto& t : lost_)    t.predict();

        // 2) asociación principal (high) con matcher ultralite
        auto m1 = match_iou_fast_(tracked_, dets_high, 0.15f);
        apply_matches_(tracked_, dets_high, m1.matches);

        // 3) segunda asociación (los no emparejados ↔ low)
        if (!m1.um_left.empty() && !dets_low.empty()){
            std::vector<Track> left; left.reserve(m1.um_left.size());
            for (int i : m1.um_left) left.push_back(tracked_[i]);
            auto m2 = match_iou_fast_(left, dets_low, 0.15f);
            for (auto& pr : m2.matches){
                int li = pr.first, di = pr.second;
                int ti = m1.um_left[li];
                tracked_[ti].update(dets_low[di]);
            }
            // reconstruye los realmente no emparejados
            std::vector<char> used(m1.um_left.size(),0);
            for (auto& pr : m2.matches) used[pr.first]=1;
            std::vector<int> rest;
            for (size_t i=0;i<m1.um_left.size();++i)
                if (!used[i]) rest.push_back(m1.um_left[i]);
            m1.um_left.swap(rest);
        }

        // 4) los que siguen sin match → Lost
        for (int i : m1.um_left){
            tracked_[i].mark_lost();
            lost_.push_back(tracked_[i]);
        }
        // compactar tracked_ (quitar movidos a lost_)
        if (!m1.um_left.empty()){
            std::vector<char> rm(tracked_.size(),0);
            for (int i : m1.um_left) if (i>=0 && i<(int)rm.size()) rm[i]=1;
            std::vector<Track> keep; keep.reserve(tracked_.size());
            for (size_t i=0;i<tracked_.size();++i) if (!rm[i]) keep.push_back(tracked_[i]);
            tracked_.swap(keep);
        }

        // 5) nuevas trayectorias por dets_high no usadas
        for (int di : m1.um_right) tracked_.emplace_back(next_id_++, dets_high[di]);

        // 6) reciclar perdidos antiguos
        std::vector<Track> keep_lost; keep_lost.reserve(lost_.size());
        for (auto& t : lost_){
            if (t.time_since_update <= max_time_lost_) keep_lost.push_back(t);
            else removed_.push_back(t);
        }
        lost_.swap(keep_lost);

        // 7) salida
        std::vector<Track> out; out.reserve(tracked_.size()+lost_.size());
        out.insert(out.end(), tracked_.begin(), tracked_.end());
        out.insert(out.end(), lost_.begin(),    lost_.end());
        return out;
    }

private:
    struct MatchRes {
        std::vector<std::pair<int,int>> matches; // (ti, di)
        std::vector<int> um_left;
        std::vector<int> um_right;
    };

    // ===== Matcher ultralite: mejor IoU por tracker (sin ordenar pares) =====
    MatchRes match_iou_fast_(const std::vector<Track>& trks,
                             const std::vector<Det>& dets,
                             float iou_gate = 0.15f)
    {
        MatchRes R;
        const int T = (int)trks.size(), D = (int)dets.size();
        if (T == 0){
            R.um_right.resize(D);
            std::iota(R.um_right.begin(), R.um_right.end(), 0);
            return R;
        }
        std::vector<char> usedT(T,0), usedD(D,0);
        for (int i=0;i<T;++i){
            float best = iou_gate; int best_j = -1;
            for (int j=0;j<D;++j){
                if (usedD[j]) continue;
                float io = iou_rect_bt(trks[i].tlwh, dets[j].tlwh);
                if (io > best){ best = io; best_j = j; }
            }
            if (best_j >= 0 && best >= iou_thr_){
                usedT[i] = usedD[best_j] = 1;
                R.matches.emplace_back(i, best_j);
            }
        }
        for (int i=0;i<T;++i) if (!usedT[i]) R.um_left.push_back(i);
        for (int j=0;j<D;++j) if (!usedD[j]) R.um_right.push_back(j);
        return R;
    }

    // Aplica matches (mínimo trabajo para velocidad)
    void apply_matches_(std::vector<Track>& trackers,
                        const std::vector<Det>& dets,
                        const std::vector<std::pair<int,int>>& matches)
    {
        for (auto& pr : matches){
            int ti = pr.first, di = pr.second;
            if (ti<0 || ti>=(int)trackers.size()) continue;
            if (di<0 || di>=(int)dets.size())      continue;
            trackers[ti].update(dets[di]);
        }
    }

private:
    int   next_id_       = 0;
    int   max_time_lost_ = 8;    // lite: elimina rápido lo perdido
    float iou_thr_       = 0.25f;

    std::vector<Track> tracked_;
    std::vector<Track> lost_;
    std::vector<Track> removed_;
};

#endif // BYTETRACK_H
