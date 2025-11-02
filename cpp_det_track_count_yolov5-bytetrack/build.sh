#!/bin/bash
set -e

g++ -O3 -std=c++14 \
  main_bytetrack.cpp yolov5_trt_bt.cpp bytetrack.cpp \
  -I. -I/usr/include/aarch64-linux-gnu -I/usr/local/cuda/include \
  `pkg-config --cflags --libs opencv4` \
  -L/usr/local/cuda/lib64 \
  -lnvinfer -lcudart -lpthread -o portal_bytetrack

echo "✅ Compilado: ./portal_bytetrack"
