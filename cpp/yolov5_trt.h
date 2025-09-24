#ifndef YOLOV5_TRT_H
#define YOLOV5_TRT_H

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>
#include <array>
#include <cstdint>

class YoLov5TRT {
public:
    explicit YoLov5TRT(const std::string& engine_path);
    ~YoLov5TRT();

    // Predicciones crudas: N x 6  (interpretación la hace main)
    std::vector<std::array<float,6>> infer(const cv::Mat& bgr);

    // Últimos parámetros de letterbox (para deshacer)
    float last_scale()    const { return lastScale; }
    int   last_pad_left() const { return lastPadLeft; }
    int   last_pad_top()  const { return lastPadTop;  }

    int out_numel() const { return outputNumel; }

private:
    bool load_engine(const std::string& path);
    void letterbox(const cv::Mat& src, cv::Mat& dst,
                   int& padLeft, int& padTop, float& scale);

    void build_lut_fp16(); // LUT 0..255 -> FP16(v/255)

private:
    nvinfer1::IRuntime*          runtime  = nullptr;
    nvinfer1::ICudaEngine*       engine   = nullptr;
    nvinfer1::IExecutionContext* context  = nullptr;

    // CUDA stream + bindings
    cudaStream_t stream = nullptr;
    void* deviceBindings[2] = {nullptr, nullptr};
    void* dInput  = nullptr;
    void* dOutput = nullptr;

    // Tipos/bytes
    nvinfer1::DataType inType;
    nvinfer1::DataType outType;
    size_t inBytes  = 0;
    size_t outBytes = 0;

    // Host staging
    // - pinned para H2D/D2H
    uint8_t* hInPinned  = nullptr;
    uint8_t* hOutPinned = nullptr;
    // - float intermedio para preproc (fallback FP32)
    std::vector<float> hInFloat;

    // LUT FP16 para normalización directa (0..255 -> half(v/255))
    std::array<uint16_t,256> lutFP16{};

    // Dimensiones
    int inputW = 640, inputH = 640;
    int inputIndex  = -1, outputIndex = -1;
    int outputNumel = 0;

    // cache letterbox
    float lastScale   = 1.f;
    int   lastPadLeft = 0;
    int   lastPadTop  = 0;
};

#endif // YOLOV5_TRT_H
