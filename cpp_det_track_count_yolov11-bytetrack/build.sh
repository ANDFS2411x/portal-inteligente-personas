#!/usr/bin/env bash
set -e
CXX=${CXX:-g++}
$CXX -O3 -std=c++14 \
  main_yolov11_bytetrack.cpp yolov11_trt.cpp \
  -I. -I/usr/local/cuda/include \
  `pkg-config --cflags --libs opencv4` \
  -L/usr/local/cuda/lib64 -lnvinfer -lcudart -lpthread \
  -o portal_yolov11_bytetrack

echo "✅ Compilado -> ./portal_yolov11_bytetrack /ruta/al/engine"
