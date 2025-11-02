#ifndef YOLOV5_TRT_H
#define YOLOV5_TRT_H

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

class YoLov5TRT {
public:
    explicit YoLov5TRT(const std::string& engine_path);
    ~YoLov5TRT();

    // Predicciones crudas: N x 6 (cx,cy,w,h,conf,cls)
    std::vector<std::array<float,6>> infer(const cv::Mat& bgr);

    // Parámetros de letterbox aplicados al último frame (para “deshacer”)
    float last_scale()    const { return lastScale; }
    int   last_pad_left() const { return lastPadLeft; }
    int   last_pad_top()  const { return lastPadTop;  }

private:
    bool load_engine(const std::string& path);
    void letterbox(const cv::Mat& src, cv::Mat& dst, int& padLeft, int& padTop, float& scale);
    void build_lut_fp16(); // LUT [0..255] -> FP16(v/255)

private:
    // TRT
    nvinfer1::IRuntime*          runtime  = nullptr;
    nvinfer1::ICudaEngine*       engine   = nullptr;
    nvinfer1::IExecutionContext* context  = nullptr;

    // CUDA
    cudaStream_t stream = nullptr;
    void* deviceBindings[2] = {nullptr, nullptr};
    void* dInput  = nullptr;
    void* dOutput = nullptr;

    // tipos/tamaños
    nvinfer1::DataType inType;
    nvinfer1::DataType outType;
    size_t inBytes  = 0;
    size_t outBytes = 0;

    // Host staging
    uint8_t*          hInPinned  = nullptr;
    uint8_t*          hOutPinned = nullptr;
    std::vector<float> hInFloat;

    // LUT 0..255 -> half(v/255)
    std::array<uint16_t,256> lutFP16{};

    // dims/letterbox
    int inputW = 640, inputH = 640;
    int inputIndex  = -1, outputIndex = -1;
    int outputNumel = 0;

    float lastScale   = 1.f;
    int   lastPadLeft = 0;
    int   lastPadTop  = 0;
};

#endif // YOLOV5_TRT_H
