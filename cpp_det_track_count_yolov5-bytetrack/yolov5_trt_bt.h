#ifndef YOLOV5_TRT_BT_H
#define YOLOV5_TRT_BT_H

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <array>
#include <string>
#include <vector>
#include <cstdint>

class YoLov5TRT_BT {
public:
    explicit YoLov5TRT_BT(const std::string& engine_path);
    ~YoLov5TRT_BT();

    // Predicciones crudas: N x 6 (cx,cy,w,h,conf,cls)
    std::vector<std::array<float,6>> infer(const cv::Mat& bgr);

    float last_scale()    const { return lastScale; }
    int   last_pad_left() const { return lastPadLeft; }
    int   last_pad_top()  const { return lastPadTop;  }

private:
    bool load_engine(const std::string& path);
    void letterbox(const cv::Mat& src, cv::Mat& dst, int& padLeft, int& padTop, float& scale);
    void build_lut_fp16();

private:
    nvinfer1::IRuntime*          runtime  = nullptr;
    nvinfer1::ICudaEngine*       engine   = nullptr;
    nvinfer1::IExecutionContext* context  = nullptr;

    cudaStream_t stream = nullptr;
    void* deviceBindings[2] = {nullptr, nullptr};
    void* dInput  = nullptr;
    void* dOutput = nullptr;

    nvinfer1::DataType inType, outType;
    size_t inBytes=0, outBytes=0;

    uint8_t* hInPinned=nullptr;
    uint8_t* hOutPinned=nullptr;
    std::vector<float> hInFloat;
    std::array<uint16_t,256> lutFP16{};

    int inputW=640, inputH=640;
    int inputIndex=-1, outputIndex=-1, outputNumel=0;

    float lastScale=1.f; int lastPadLeft=0, lastPadTop=0;
};

#endif // YOLOV5_TRT_BT_H
