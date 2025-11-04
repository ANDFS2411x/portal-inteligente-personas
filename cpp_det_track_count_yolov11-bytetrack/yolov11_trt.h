#ifndef YOLOV11_TRT_H
#define YOLOV11_TRT_H

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>
#include <array>
#include <cstdint>

class YoLov11TRT {
public:
    explicit YoLov11TRT(const std::string& engine_path);
    ~YoLov11TRT();

    // Predicciones crudas: N x 5  [cx,cy,w,h,conf] (una sola clase)
    std::vector<std::array<float,5>> infer(const cv::Mat& bgr);

    // Últimos parámetros de letterbox (para deshacer)
    float last_scale()    const { return lastScale; }
    int   last_pad_left() const { return lastPadLeft; }
    int   last_pad_top()  const { return lastPadTop;  }

private:
    bool load_engine_(const std::string& path);
    void letterbox_(const cv::Mat& src, cv::Mat& dst,
                    int& padLeft, int& padTop, float& scale);
    void build_lut_fp16_();

private:
    nvinfer1::IRuntime*          runtime  = nullptr;
    nvinfer1::ICudaEngine*       engine   = nullptr;
    nvinfer1::IExecutionContext* context  = nullptr;

    cudaStream_t stream = nullptr;
    void* dInput  = nullptr;
    void* dOutput = nullptr;
    void* deviceBindings[2] = {nullptr, nullptr};

    int inputIndex  = -1;
    int outputIndex = -1;

    nvinfer1::DataType inType;
    nvinfer1::DataType outType;

    size_t inBytes  = 0;
    size_t outBytes = 0;
    int outputNumel = 0;

    // Host staging
    uint8_t* hInPinned  = nullptr;
    uint8_t* hOutPinned = nullptr;
    std::vector<float> hInFloat;

    // LUT 0..255 -> half(v/255)
    std::array<uint16_t,256> lutFP16{};

    // Dimensiones esperadas
    int inputW = 640, inputH = 640;

    // cache letterbox
    float lastScale   = 1.f;
    int   lastPadLeft = 0;
    int   lastPadTop  = 0;
};

#endif // YOLOV11_TRT_H
