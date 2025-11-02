#include "yolov5_trt_bt.h"
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>

// ===== Logger TRT =====
class LoggerBT : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cout << "[TensorRT] " << msg << std::endl;
    }
};
static LoggerBT gLoggerBT;

// ===== util half <-> float =====
static inline float half_to_float_bt(uint16_t h){
    uint32_t sign=(h&0x8000)<<16, exp=(h&0x7C00)>>10, mant=(h&0x03FF), f;
    if(exp==0){ if(mant==0) f=sign; else{ exp=127-15+1; while((mant&0x0400)==0){mant<<=1; exp--;} mant&=0x03FF; f=sign | (exp<<23) | (mant<<13);} }
    else if(exp==0x1F){ f=sign | 0x7F800000 | (mant<<13); }
    else { uint32_t e=exp-15+127; f=sign | (e<<23) | (mant<<13); }
    float out; std::memcpy(&out,&f,sizeof(out)); return out;
}
static inline uint16_t float_to_half_bits_bt(float f){
    uint32_t x; std::memcpy(&x,&f,sizeof(x));
    uint32_t sign=(x>>16)&0x8000; int32_t exp=((x>>23)&0xFF)-127+15; uint32_t mant=x&0x7FFFFF;
    if(exp<=0){ if(exp<-10) return (uint16_t)sign; mant=(mant|0x800000)>>(1-exp); return (uint16_t)(sign | (mant+0x1000)>>13); }
    else if(exp>=31){ return (uint16_t)(sign | 0x7C00); }
    else { return (uint16_t)(sign | (exp<<10) | ((mant+0x1000)>>13)); }
}
static inline size_t dataTypeSizeBT(nvinfer1::DataType t){
    using DT=nvinfer1::DataType;
    switch(t){ case DT::kFLOAT: return 4; case DT::kHALF: return 2; case DT::kINT8: return 1; case DT::kINT32: return 4; case DT::kBOOL: return 1; default: return 4; }
}
static inline int64_t volumeBT(const nvinfer1::Dims& d){ int64_t v=1; for(int i=0;i<d.nbDims;++i) v*=std::max(1,d.d[i]); return v; }

// ===== ctor/dtor =====
YoLov5TRT_BT::YoLov5TRT_BT(const std::string& engine_path){
    if(!load_engine(engine_path)){ std::cerr<<"❌ No se pudo cargar el engine\n"; std::exit(1); }
}
YoLov5TRT_BT::~YoLov5TRT_BT(){
    if(hInPinned)  cudaFreeHost(hInPinned);
    if(hOutPinned) cudaFreeHost(hOutPinned);
    if(dInput)  cudaFree(dInput);
    if(dOutput) cudaFree(dOutput);
    if(stream)  cudaStreamDestroy(stream);
    if(context) context->destroy(); // warnings ok
    if(engine)  engine->destroy();
    if(runtime) runtime->destroy();
}

// ===== engine =====
bool YoLov5TRT_BT::load_engine(const std::string& path){
    std::ifstream f(path, std::ios::binary); if(!f.good()) return false;
    f.seekg(0,f.end); size_t size=f.tellg(); f.seekg(0,f.beg);
    std::vector<char> data(size); f.read(data.data(), size); f.close();

    runtime = nvinfer1::createInferRuntime(gLoggerBT);
    engine  = runtime->deserializeCudaEngine(data.data(), size);
    if(!engine) return false;
    context = engine->createExecutionContext();
    if(!context) return false;

    int nb=engine->getNbBindings();
    for(int i=0;i<nb;++i){
        if(engine->bindingIsInput(i)) inputIndex=i; else outputIndex=i;
        std::cout << (engine->bindingIsInput(i)?"Input ":"Output ") << i << ": " << engine->getBindingName(i) << "\n";
    }
    inType  = engine->getBindingDataType(inputIndex);
    outType = engine->getBindingDataType(outputIndex);

    auto outDims = context->getBindingDimensions(outputIndex);
    outputNumel = (int)volumeBT(outDims);
    outBytes    = outputNumel * dataTypeSizeBT(outType);

    auto inDims = context->getBindingDimensions(inputIndex);
    int64_t inNumel = volumeBT(inDims);
    inBytes = inNumel * dataTypeSizeBT(inType);

    hInFloat.resize(inNumel, 0.f);
    cudaHostAlloc((void**)&hInPinned,  inBytes,  cudaHostAllocDefault);
    cudaHostAlloc((void**)&hOutPinned, outBytes, cudaHostAllocDefault);
    cudaMalloc(&dInput,  inBytes);
    cudaMalloc(&dOutput, outBytes);

    deviceBindings[inputIndex]=dInput;
    deviceBindings[outputIndex]=dOutput;

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    build_lut_fp16();
    return true;
}

void YoLov5TRT_BT::build_lut_fp16(){ for(int i=0;i<256;++i){ float f=i/255.f; lutFP16[i]=float_to_half_bits_bt(f);} }

// ===== letterbox =====
void YoLov5TRT_BT::letterbox(const cv::Mat& src, cv::Mat& dst, int& padLeft, int& padTop, float& scale){
    int w=src.cols,h=src.rows; scale=std::min(640.f/w, 640.f/h);
    int newW=std::round(w*scale), newH=std::round(h*scale);
    int padW=640-newW, padH=640-newH;
    padLeft=padW/2; padTop=padH/2;
    if(newW!=w || newH!=h) cv::resize(src,dst,cv::Size(newW,newH)); else dst=src;
    if(padW||padH) cv::copyMakeBorder(dst,dst,padTop,padH-padTop,padLeft,padW-padLeft,cv::BORDER_CONSTANT,{114,114,114});
    lastScale=scale; lastPadLeft=padLeft; lastPadTop=padTop;
}

// ===== infer =====
std::vector<std::array<float,6>> YoLov5TRT_BT::infer(const cv::Mat& bgr){
    cv::Mat lb; int pl,pt; float sc; letterbox(bgr, lb, pl, pt, sc);

    if (inType==nvinfer1::DataType::kHALF){
        cv::Mat rgb8; cv::cvtColor(lb, rgb8, cv::COLOR_BGR2RGB);
        const int HW=640*640; auto* hp=reinterpret_cast<uint16_t*>(hInPinned);
        uint16_t *c0=hp+0*HW, *c1=hp+1*HW, *c2=hp+2*HW;
        for(int y=0;y<640;++y){ const cv::Vec3b* row=rgb8.ptr<cv::Vec3b>(y); int base=y*640;
            for(int x=0;x<640;++x){ const cv::Vec3b& v=row[x]; c0[base+x]=lutFP16[v[0]]; c1[base+x]=lutFP16[v[1]]; c2[base+x]=lutFP16[v[2]]; } }
    } else {
        cv::Mat rgb; cv::cvtColor(lb, rgb, cv::COLOR_BGR2RGB); rgb.convertTo(rgb, CV_32F, 1.f/255.f);
        std::vector<cv::Mat> ch(3); cv::split(rgb,ch);
        float* p=hInFloat.data();
        std::memcpy(p, ch[0].data, 640*640*sizeof(float));
        std::memcpy(p+640*640, ch[1].data, 640*640*sizeof(float));
        std::memcpy(p+2*640*640, ch[2].data, 640*640*sizeof(float));
        std::memcpy(hInPinned, hInFloat.data(), hInFloat.size()*sizeof(float));
    }

    cudaMemcpyAsync(dInput, hInPinned, inBytes, cudaMemcpyHostToDevice, stream);
    context->enqueueV2(deviceBindings, stream, nullptr);
    cudaMemcpyAsync(hOutPinned, dOutput, outBytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int outNumel = outBytes / (outType==nvinfer1::DataType::kHALF?2:4);
    std::vector<float> outFloat(outNumel);
    if (outType==nvinfer1::DataType::kHALF){
        auto* src=reinterpret_cast<const uint16_t*>(hOutPinned);
        for(int i=0;i<outNumel;++i) outFloat[i]=half_to_float_bt(src[i]);
    } else {
        std::memcpy(outFloat.data(), hOutPinned, outBytes);
    }

    const int COLS=6; int N=outNumel/COLS;
    std::vector<std::array<float,6>> preds(N);
    for(int i=0;i<N;++i) for(int j=0;j<COLS;++j) preds[i][j]=outFloat[i*COLS+j];
    return preds;
}
