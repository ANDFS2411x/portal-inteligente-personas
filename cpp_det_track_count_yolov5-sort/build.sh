#!/bin/bash
set -e

g++ -O3 -std=c++14   main.cpp yolov5_trt.cpp   -I. -I/usr/include/aarch64-linux-gnu -I/usr/local/cuda/include   `pkg-config --cflags --libs opencv4`   -L/usr/local/cuda/lib64   -lnvinfer -lcudart -lpthread -o portal
echo "✅ Compilado: ./portal"
