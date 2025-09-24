#include "yolov5_trt.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <string>

// ================= Config rápida =================
static const bool SHOW = true;   // pon false para medir FPS máximo del engine
static const bool SAVE = true;   // true para grabar detección en AVI
// Si quieres probar GStreamer con nvjpegdec + nvvidconv a 640x640, pon true
static const bool USE_GSTREAMER = false;

// clamp C++11
template<typename T>
static inline T clamp_(T v, T lo, T hi) { return v < lo ? lo : (v > hi ? hi : v); }

// IoU + NMS
static float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    float uni   = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.f;
}
static std::vector<int> nms(const std::vector<cv::Rect2f>& boxes,
                            const std::vector<float>& scores, float iouTh) {
    std::vector<int> order(boxes.size()); std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){ return scores[a] > scores[b]; });
    std::vector<int> keep; std::vector<char> rem(boxes.size(), 0);
    for (size_t m=0; m<order.size(); ++m) {
        int i = order[m]; if (rem[i]) continue; keep.push_back(i);
        for (size_t n=m+1; n<order.size(); ++n) {
            int j = order[n]; if (rem[j]) continue;
            if (iou(boxes[i], boxes[j]) > iouTh) rem[j] = 1;
        }
    }
    return keep;
}

int main() {
    const std::string engine_path = "/home/andfs/portal-inteligente-personas/best4n_100epocas.engine";

    YoLov5TRT detector(engine_path);

    cv::VideoCapture cap;
    if (USE_GSTREAMER) {
        // Prueba pipeline HW: decodifica MJPEG + reescala a 640x640 en GPU/VIC
        std::string pipe =
            "v4l2src device=/dev/video0 ! "
            "image/jpeg,framerate=30/1,width=640,height=480 ! "
            "jpegparse ! nvjpegdec ! "
            "nvvidconv ! video/x-raw,format=BGR,width=640,height=640 ! "
            "appsink drop=true max-buffers=1";
        cap.open(pipe, cv::CAP_GSTREAMER);
    } else {
        // V4L2 directo (rápido y estable en Nano)
        cap.open(0, cv::CAP_V4L2);
        if (cap.isOpened()) {
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
            cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        }
    }

    if (!cap.isOpened()) {
        std::cerr << "❌ No se pudo abrir la cámara\n";
        return -1;
    }

    int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    if (W <= 0 || H <= 0) { W = 640; H = USE_GSTREAMER ? 640 : 480; }

    cv::VideoWriter writer;
    if (SAVE) {
        writer.open("deteccion_output.avi",
            cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(W, H));
        if (!writer.isOpened()) {
            std::cerr << "⚠️ No se pudo abrir el VideoWriter, se desactiva SAVE.\n";
        }
    }

    std::ofstream csv("metrics_log.csv");
    csv << "frame_id,timestamp,fps,n_dets,mean_conf\n";

    int frame_id = 0;
    const float CONF_THR = 0.30f;
    const float NMS_IOU  = 0.50f;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto raw = detector.infer(frame); // N x 6
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms  = std::chrono::duration<float, std::milli>(t1 - t0).count();
        float fps = ms > 0 ? 1000.0f / ms : 0.0f;

        // Deshacer letterbox
        float r  = detector.last_scale();
        int   dl = detector.last_pad_left();
        int   dt = detector.last_pad_top();

        std::vector<cv::Rect2f> boxes; std::vector<float> scores;

        // Decodificador cxcywh (como en tu Python)
        for (const auto& det : raw) {
            float cx=det[0], cy=det[1], w=det[2], h=det[3], conf=det[4];
            // int   cls = (int)det[5]; // si tu clase persona no es 0, no filtres por clase
            if (!std::isfinite(conf) || conf < CONF_THR) continue;

            float x1l = cx - w*0.5f, y1l = cy - h*0.5f;
            float x2l = cx + w*0.5f, y2l = cy + h*0.5f;

            float x1 = (x1l - dl) / r;
            float y1 = (y1l - dt) / r;
            float x2 = (x2l - dl) / r;
            float y2 = (y2l - dt) / r;

            x1 = clamp_(x1, 0.f, (float)W-1);
            y1 = clamp_(y1, 0.f, (float)H-1);
            x2 = clamp_(x2, 0.f, (float)W-1);
            y2 = clamp_(y2, 0.f, (float)H-1);
            if (x2<=x1 || y2<=y1) continue;

            boxes.emplace_back(x1, y1, x2-x1, y2-y1);
            scores.push_back(conf);
        }

        // NMS
        std::vector<int> keep;
        if (!boxes.empty()) keep = nms(boxes, scores, NMS_IOU);

        // Dibujar + métricas
        int n_dets = (int)keep.size();
        float mean_conf = 0.f;
        if (SHOW || SAVE) {
            for (int k : keep) {
                mean_conf += scores[k];
                cv::rectangle(frame, boxes[k], {255,0,0}, 2);
                // Para apurar FPS, deja sin texto; si quieres, descomenta:
                // cv::putText(frame, cv::format("%.2f", scores[k]),
                //             {(int)boxes[k].x, std::max(0, (int)boxes[k].y - 5)},
                //             cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,255,255}, 2);
            }
        }
        if (n_dets > 0) mean_conf /= n_dets;

        if (SHOW) {
            cv::putText(frame, cv::format("FPS: %.1f", fps), {10,30},
                        cv::FONT_HERSHEY_SIMPLEX, 1, {0,255,0}, 2);
            cv::imshow("Detecciones", frame);
        }
        if (SAVE && writer.isOpened()) writer.write(frame);

        csv << frame_id++ << ","
            << std::chrono::duration_cast<std::chrono::milliseconds>(t1.time_since_epoch()).count() << ","
            << fps << "," << n_dets << "," << (n_dets ? mean_conf : 0.0f) << "\n";

        if (SHOW) {
            int k = cv::waitKey(1);
            if (k == 27 || k == 'q') break;
        }
    }

    cap.release();
    if (SAVE && writer.isOpened()) writer.release();
    csv.close();
    if (SHOW) cv::destroyAllWindows();
    return 0;
}
