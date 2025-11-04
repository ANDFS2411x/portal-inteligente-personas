#!/usr/bin/env bash
set -e

CXX=${CXX:-g++}
$CXX -O3 -std=c++14 \
  main_yolov11_sort.cpp yolov11_trt.cpp \
  -I. -I/usr/include/aarch64-linux-gnu -I/usr/local/cuda/include \
  `pkg-config --cflags --libs opencv4` \
  -L/usr/local/cuda/lib64 \
  -lnvinfer -lcudart -lpthread -o portal_yolov11_sort

echo "✅ build OK -> ./portal_yolov11_sort /ruta/al/engine"
