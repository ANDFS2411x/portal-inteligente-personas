#include "yolov11_trt.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cassert>

// ===== Utils FP16 <-> FP32 host =====
static inline float half_to_float(uint16_t h){
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);
    uint32_t f;
    if (exp == 0){
        if (mant == 0) f = sign;
        else{
            exp = 127 - 15 + 1;
            while ((mant & 0x0400) == 0){ mant <<= 1; exp--; }
            mant &= 0x03FF;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F){
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        uint32_t e = exp - 15 + 127;
        f = sign | (e << 23) | (mant << 13);
    }
    float out; std::memcpy(&out, &f, sizeof(out)); return out;
}
static inline uint16_t float_to_half_bits(float f){
    uint32_t x; std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0){
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000) >> (1 - exp);
        return (uint16_t)(sign | (mant + 0x1000) >> 13);
    } else if (exp >= 31){
        return (uint16_t)(sign | 0x7C00);
    } else {
        return (uint16_t)(sign | (exp << 10) | ((mant + 0x1000) >> 13));
    }
}
static inline size_t dt_size(nvinfer1::DataType t){
    using DT = nvinfer1::DataType;
    switch(t){
        case DT::kFLOAT: return 4;
        case DT::kHALF:  return 2;
        case DT::kINT8:  return 1;
        case DT::kINT32: return 4;
        case DT::kBOOL:  return 1;
        default:         return 4;
    }
}
static inline int64_t volume(const nvinfer1::Dims& d){
    int64_t v = 1;
    for (int i=0;i<d.nbDims;++i) v *= std::max(1, d.d[i]);
    return v;
}

// ===== Logger TRT =====
class Logger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cout << "[TensorRT] " << msg << std::endl;
    }
};
static Logger gLogger;

// ===== Ctor/Dtor =====
YoLov11TRT::YoLov11TRT(const std::string& engine_path){
    if (!load_engine_(engine_path)){
        std::cerr << "❌ No se pudo cargar el engine: " << engine_path << "\n";
        std::exit(1);
    }
}
YoLov11TRT::~YoLov11TRT(){
    if (hInPinned)  cudaFreeHost(hInPinned);
    if (hOutPinned) cudaFreeHost(hOutPinned);
    if (dInput)  cudaFree(dInput);
    if (dOutput) cudaFree(dOutput);
    if (stream)  cudaStreamDestroy(stream);
    if (context) context->destroy();  // TRT8 avisa deprecated; ok
    if (engine)  engine->destroy();
    if (runtime) runtime->destroy();
}

// ===== Engine load =====
bool YoLov11TRT::load_engine_(const std::string& path){
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    f.seekg(0, f.end); size_t size = f.tellg(); f.seekg(0, f.beg);
    std::vector<char> data(size);
    f.read(data.data(), size); f.close();

    runtime = nvinfer1::createInferRuntime(gLogger);
    engine  = runtime->deserializeCudaEngine(data.data(), size);
    if (!engine) return false;
    context = engine->createExecutionContext();
    if (!context) return false;

    // Bindings esperados: "images" (input), "output0" (1,5,8400)
    inputIndex  = engine->getBindingIndex("images");
    outputIndex = engine->getBindingIndex("output0");
    if (inputIndex < 0 || outputIndex < 0){
        std::cerr << "❌ No se encontraron bindings 'images'/'output0'\n";
        return false;
    }

    inType  = engine->getBindingDataType(inputIndex);
    outType = engine->getBindingDataType(outputIndex);

    auto outDims = context->getBindingDimensions(outputIndex);
    outputNumel  = static_cast<int>(volume(outDims));
    outBytes     = outputNumel * dt_size(outType);

    auto inDims  = context->getBindingDimensions(inputIndex);
    int64_t inNumel = volume(inDims);
    inBytes = inNumel * dt_size(inType);

    // Reservas
    hInFloat.resize(inNumel, 0.f);
    cudaHostAlloc((void**)&hInPinned,  inBytes,  cudaHostAllocDefault);
    cudaHostAlloc((void**)&hOutPinned, outBytes, cudaHostAllocDefault);
    cudaMalloc(&dInput,  inBytes);
    cudaMalloc(&dOutput, outBytes);
    deviceBindings[inputIndex]  = dInput;
    deviceBindings[outputIndex] = dOutput;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    build_lut_fp16_();
    return true;
}

void YoLov11TRT::build_lut_fp16_(){
    for (int i=0;i<256;++i){
        float f = i / 255.0f;
        lutFP16[i] = float_to_half_bits(f);
    }
}

// ===== Letterbox =====
void YoLov11TRT::letterbox_(const cv::Mat& src, cv::Mat& dst,
                            int& padLeft, int& padTop, float& scale){
    int w = src.cols, h = src.rows;
    scale = std::min((float)inputW / w, (float)inputH / h);
    int newW = std::round(w * scale);
    int newH = std::round(h * scale);
    int padW = inputW - newW;
    int padH = inputH - newH;
    padLeft = padW / 2;
    padTop  = padH / 2;

    if (newW != w || newH != h) cv::resize(src, dst, cv::Size(newW, newH));
    else dst = src;
    if (padW != 0 || padH != 0){
        cv::copyMakeBorder(dst, dst, padTop, padH - padTop, padLeft, padW - padLeft,
                           cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
    }
    lastScale   = scale;
    lastPadLeft = padLeft;
    lastPadTop  = padTop;
}

// ===== Infer (FP16/FP32 compatibles) =====
std::vector<std::array<float,5>> YoLov11TRT::infer(const cv::Mat& bgr){
    // 1) Preprocess con letterbox a 640x640
    cv::Mat lb; int pl, pt; float sc;
    letterbox_(bgr, lb, pl, pt, sc);

    // 2) Cargar input en layout CHW y normalizar 0..1
    if (inType == nvinfer1::DataType::kHALF){
        cv::Mat rgb8; cv::cvtColor(lb, rgb8, cv::COLOR_BGR2RGB);
        const int HW = inputW * inputH;
        auto* hp = reinterpret_cast<uint16_t*>(hInPinned);
        uint16_t* c0 = hp + 0*HW; // R
        uint16_t* c1 = hp + 1*HW; // G
        uint16_t* c2 = hp + 2*HW; // B
        for (int y=0; y<inputH; ++y){
            const cv::Vec3b* row = rgb8.ptr<cv::Vec3b>(y);
            int base = y * inputW;
            for (int x=0; x<inputW; ++x){
                const cv::Vec3b& v = row[x]; // (R,G,B)
                c0[base+x] = lutFP16[v[0]];
                c1[base+x] = lutFP16[v[1]];
                c2[base+x] = lutFP16[v[2]];
            }
        }
    } else {
        cv::Mat rgb; cv::cvtColor(lb, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0/255.0);
        std::vector<cv::Mat> ch(3); cv::split(rgb, ch);
        float* p = hInFloat.data();
        std::memcpy(p,                ch[0].data, inputW*inputH*sizeof(float));
        std::memcpy(p+inputW*inputH,  ch[1].data, inputW*inputH*sizeof(float));
        std::memcpy(p+2*inputW*inputH,ch[2].data, inputW*inputH*sizeof(float));
        std::memcpy(hInPinned, hInFloat.data(), hInFloat.size()*sizeof(float));
    }

    // 3) H2D + enqueue + D2H
    cudaMemcpyAsync(dInput, hInPinned, inBytes, cudaMemcpyHostToDevice, stream);
    context->enqueueV2(deviceBindings, stream, nullptr);
    cudaMemcpyAsync(hOutPinned, dOutput, outBytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 4) Convertir salida a float32
    std::vector<float> outFloat(outputNumel);
    if (outType == nvinfer1::DataType::kHALF){
        auto* src = reinterpret_cast<const uint16_t*>(hOutPinned);
        for (int i=0;i<outputNumel;++i) outFloat[i] = half_to_float(src[i]);
    } else {
        std::memcpy(outFloat.data(), hOutPinned, outBytes);
    }

    // 5) Reempaquetar: salida (1,5,8400) -> vector<array<float,5>> con orden [cx,cy,w,h,conf]
    // Nota: muchos exports de YOLOv11 usan layout [C, Npred], donde C=5
    // Vamos a detectar posible layout [1,5,8400] vs [1,8400,5]. En TRT suele ser [1,5,8400].
    const int C = 5;
    int N = outputNumel / C;
    std::vector<std::array<float,5>> preds; preds.resize(N);

    // Intento con layout [1, C, N]
    bool assumed_CHW = true;
    // chequeo simple: valores de conf deben estar en [0..1] razonable
    // probamos leyendo conf como out[4*N + i] para algunos i
    int probe = std::min(N, 10);
    int ok = 0;
    for (int i=0;i<probe;++i){
        float conf = outFloat[4*N + i];
        if (conf >= 0.f && conf <= 1.5f) ok++;
    }
    if (ok < probe/2) assumed_CHW = false;

    if (assumed_CHW){
        // [1, 5, N] -> channel-major
        const float* cxp = outFloat.data() + 0*N;
        const float* cyp = outFloat.data() + 1*N;
        const float* wp  = outFloat.data() + 2*N;
        const float* hp  = outFloat.data() + 3*N;
        const float* sc  = outFloat.data() + 4*N;
        for (int i=0;i<N;++i){
            preds[i][0] = cxp[i];
            preds[i][1] = cyp[i];
            preds[i][2] = wp[i];
            preds[i][3] = hp[i];
            preds[i][4] = sc[i];
        }
    } else {
        // [1, N, 5] -> row-major por pred
        for (int i=0;i<N;++i){
            preds[i][0] = outFloat[i*C + 0];
            preds[i][1] = outFloat[i*C + 1];
            preds[i][2] = outFloat[i*C + 2];
            preds[i][3] = outFloat[i*C + 3];
            preds[i][4] = outFloat[i*C + 4];
        }
    }
    return preds;
}
