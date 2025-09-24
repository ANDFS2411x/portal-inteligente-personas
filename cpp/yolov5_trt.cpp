#include "yolov5_trt.h"
#include <fstream>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cstring>

// -------- util half <-> float (host) --------
static inline float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            exp = 127 - 15 + 1;
            while ((mant & 0x0400) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FF;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        uint32_t e = exp - 15 + 127;
        f = sign | (e << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}
static inline uint16_t float_to_half_bits(float f) {
    uint32_t x; std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000) >> (1 - exp);
        return (uint16_t)(sign | (mant + 0x1000) >> 13);
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00);
    } else {
        return (uint16_t)(sign | (exp << 10) | ((mant + 0x1000) >> 13));
    }
}
static inline size_t dataTypeSize(nvinfer1::DataType t) {
    using DT = nvinfer1::DataType;
    switch (t) {
        case DT::kFLOAT: return 4;
        case DT::kHALF:  return 2;
        case DT::kINT8:  return 1;
        case DT::kINT32: return 4;
        case DT::kBOOL:  return 1;
        default:         return 4;
    }
}
static inline int64_t volume(const nvinfer1::Dims& d) {
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= std::max(1, d.d[i]);
    return v;
}

// -------- Logger --------
class Logger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cout << "[TensorRT] " << msg << std::endl;
    }
};
static Logger gLogger;

// -------- Ctor/Dtor --------
YoLov5TRT::YoLov5TRT(const std::string& engine_path) {
    if (!load_engine(engine_path)) {
        std::cerr << "❌ No se pudo cargar el engine.\n";
        std::exit(1);
    }
}
YoLov5TRT::~YoLov5TRT() {
    if (hInPinned)  cudaFreeHost(hInPinned);
    if (hOutPinned) cudaFreeHost(hOutPinned);
    if (dInput)  cudaFree(dInput);
    if (dOutput) cudaFree(dOutput);
    if (stream)  cudaStreamDestroy(stream);
    if (context) context->destroy();
    if (engine)  engine->destroy();
    if (runtime) runtime->destroy();
}

// -------- Engine --------
bool YoLov5TRT::load_engine(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    f.seekg(0, f.end);
    size_t size = f.tellg();
    f.seekg(0, f.beg);

    std::vector<char> data(size);
    f.read(data.data(), size);
    f.close();

    runtime = nvinfer1::createInferRuntime(gLogger);
    engine  = runtime->deserializeCudaEngine(data.data(), size);
    if (!engine) return false;
    context = engine->createExecutionContext();
    if (!context) return false;

    std::cout << "=== Bindings del engine ===\n";
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        std::cout << (engine->bindingIsInput(i) ? "Input" : "Output")
                  << " binding " << i << ": " << engine->getBindingName(i) << "\n";
    }

    inputIndex  = engine->getBindingIndex("images");
    outputIndex = engine->getBindingIndex("output0");
    if (inputIndex < 0 || outputIndex < 0) {
        std::cerr << "❌ No se encontraron bindings 'images'/'output0'\n";
        return false;
    }

    inType  = engine->getBindingDataType(inputIndex);
    outType = engine->getBindingDataType(outputIndex);

    auto outDims = context->getBindingDimensions(outputIndex);
    outputNumel  = static_cast<int>(volume(outDims));
    outBytes     = outputNumel * dataTypeSize(outType);

    auto inDims  = context->getBindingDimensions(inputIndex);
    int64_t inNumel = volume(inDims);
    inBytes = inNumel * dataTypeSize(inType);

    std::cout << "🔎 OUT dims: ";
    for (int i=0;i<outDims.nbDims;++i) std::cout << outDims.d[i] << " ";
    std::cout << " | dtype=" << (outType==nvinfer1::DataType::kHALF?"FP16":"FP32") << "\n";
    std::cout << "🔎 IN  dims: ";
    for (int i=0;i<inDims.nbDims;++i) std::cout << inDims.d[i] << " ";
    std::cout << " | dtype=" << (inType==nvinfer1::DataType::kHALF?"FP16":"FP32") << "\n";

    // Reservas
    hInFloat.resize(inNumel, 0.f);
    cudaHostAlloc((void**)&hInPinned,  inBytes,  cudaHostAllocDefault);
    cudaHostAlloc((void**)&hOutPinned, outBytes, cudaHostAllocDefault);
    cudaMalloc(&dInput,  inBytes);
    cudaMalloc(&dOutput, outBytes);

    deviceBindings[inputIndex]  = dInput;
    deviceBindings[outputIndex] = dOutput;

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // LUT para preproc rápido si la entrada es FP16
    build_lut_fp16();

    return true;
}

void YoLov5TRT::build_lut_fp16() {
    for (int i = 0; i < 256; ++i) {
        float f = i / 255.0f;
        lutFP16[i] = float_to_half_bits(f);
    }
}

// -------- Letterbox --------
void YoLov5TRT::letterbox(const cv::Mat& src, cv::Mat& dst,
                          int& padLeft, int& padTop, float& scale) {
    int w = src.cols, h = src.rows;
    scale = std::min((float)inputW / w, (float)inputH / h);
    int newW = std::round(w * scale);
    int newH = std::round(h * scale);
    int padW = inputW - newW;
    int padH = inputH - newH;
    padLeft = padW / 2;
    padTop  = padH / 2;

    if (newW != w || newH != h) {
        cv::resize(src, dst, cv::Size(newW, newH));
    } else {
        dst = src;
    }
    if (padW != 0 || padH != 0) {
        cv::copyMakeBorder(dst, dst, padTop, padH - padTop, padLeft, padW - padLeft,
                           cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
    }

    lastScale   = scale;
    lastPadLeft = padLeft;
    lastPadTop  = padTop;
}

// -------- Infer (async + pinned + LUT FP16) --------
std::vector<std::array<float,6>> YoLov5TRT::infer(const cv::Mat& bgr) {
    // 1) Preprocess
    cv::Mat lb; int pl, pt; float sc;
    letterbox(bgr, lb, pl, pt, sc);

    // === Camino rápido: entrada FP16 -> RGB8 + LUT -> CHW FP16 directo ===
    if (inType == nvinfer1::DataType::kHALF) {
        cv::Mat rgb8; cv::cvtColor(lb, rgb8, cv::COLOR_BGR2RGB);

        const int HW = inputW * inputH;
        auto* hp = reinterpret_cast<uint16_t*>(hInPinned);
        uint16_t* c0 = hp + 0*HW; // R
        uint16_t* c1 = hp + 1*HW; // G
        uint16_t* c2 = hp + 2*HW; // B
        for (int y = 0; y < inputH; ++y) {
            const cv::Vec3b* row = rgb8.ptr<cv::Vec3b>(y);
            int base = y * inputW;
            for (int x = 0; x < inputW; ++x) {
                const cv::Vec3b& v = row[x]; // (R,G,B)
                c0[base + x] = lutFP16[v[0]];
                c1[base + x] = lutFP16[v[1]];
                c2[base + x] = lutFP16[v[2]];
            }
        }
    } else {
        // === Fallback: entrada FP32 (normalización en float) ===
        cv::Mat rgb; cv::cvtColor(lb, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0/255.0);
        std::vector<cv::Mat> ch(3);
        cv::split(rgb, ch);
        float* p = hInFloat.data();
        std::memcpy(p,                ch[0].data, inputW*inputH*sizeof(float));
        std::memcpy(p+inputW*inputH,  ch[1].data, inputW*inputH*sizeof(float));
        std::memcpy(p+2*inputW*inputH,ch[2].data, inputW*inputH*sizeof(float));
        std::memcpy(hInPinned, hInFloat.data(), hInFloat.size()*sizeof(float));
    }

    // 2) H2D + enqueue + D2H (todo async en stream)
    cudaMemcpyAsync(dInput, hInPinned, inBytes, cudaMemcpyHostToDevice, stream);
    context->enqueueV2(deviceBindings, stream, nullptr);
    cudaMemcpyAsync(hOutPinned, dOutput, outBytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 3) Salida a float32
    std::vector<float> outFloat(outputNumel);
    if (outType == nvinfer1::DataType::kHALF) {
        auto* src = reinterpret_cast<const uint16_t*>(hOutPinned);
        for (int i=0;i<outputNumel;++i) outFloat[i] = half_to_float(src[i]);
    } else {
        std::memcpy(outFloat.data(), hOutPinned, outBytes);
    }

    // 4) Empaquetar N x 6
    const int COLS = 6;
    int N = outputNumel / COLS;
    std::vector<std::array<float,6>> preds; preds.resize(N);
    for (int i=0;i<N;++i)
        for (int j=0;j<COLS;++j)
            preds[i][j] = outFloat[i*COLS + j];
    return preds;
}
