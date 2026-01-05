<div align="center">

# Intelligent People-Counting Portal 🚪📊

[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Jetson Nano](https://img.shields.io/badge/Platform-NVIDIA_Jetson_Nano-76B900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/embedded/jetson-nano)
[![YOLO](https://img.shields.io/badge/Model-YOLOv5s-00FFFF?style=for-the-badge&logo=opencv)](https://ultralytics.com/yolo)
[![TensorRT](https://img.shields.io/badge/Optimized-TensorRT_FP16-76B900?style=for-the-badge)](https://developer.nvidia.com/tensorrt)

**An Intelligent Portal System for People Counting** using computer vision and deep learning, designed for **Edge Computing** on an NVIDIA Jetson Nano.

This system enables **precise, real-time, bidirectional counting** (entries and exits) **without cloud dependency**, ensuring low latency and complete data privacy.

</div>

---

<div align="center" style="color: #76B900; font-size: 1.4em; font-weight: bold;">
📌 Project Overview
</div>

This project implements an **overhead (top-down) vision system** to monitor occupant flow through access points. By processing video frames locally, the system detects individuals, tracks their movement, and records crossings through a virtual line.

### ✨ Key Goals
- <span style="color: #76B900;">✅</span> **Real-time Detection:** Leveraging YOLO models optimized for embedded hardware  
- <span style="color: #76B900;">✅</span> **Local Processing:** 100% offline operation on NVIDIA Jetson Nano  
- <span style="color: #76B900;">✅</span> **High Performance:** Optimized inference using NVIDIA TensorRT (FP16)  
- <span style="color: #76B900;">✅</span> **Data Privacy:** No video data leaves the local device

---

<div align="center" style="color: #76B900; font-size: 1.4em; font-weight: bold;">
🧠 System Architecture
</div>

### 1️⃣ General System Diagram
The complete flow from video acquisition and frame pre-processing to visualization.

<img src="images/system_diagram.png" alt="General System Diagram" width="100%"/>

### 2️⃣ Block Diagram
Internal component interaction and data flow optimized for the Jetson Nano environment.

<img src="images/block_diagram.png" alt="Block Diagram" width="100%"/>

---

<div align="center" style="color: #76B900; font-size: 1.4em; font-weight: bold;">
🔄 Optimization & Multithreading
</div>

To maintain a stable frame rate on low-resource hardware, the system utilizes a **three-thread architecture**:

1. **Capture Thread** — Frame acquisition via GStreamer  
2. **Processing Thread** — Inference (TensorRT), Tracking (SORT), and counting logic  
3. **Visualization Thread** — Real-time UI rendering and statistics display

### Model Conversion Pipeline (.pt → .onnx → .engine)
Deep learning models are converted to **TensorRT Engines with FP16 precision** to maximize CUDA core utilization.

<img src="images/model_conversion.png" alt="Model Conversion Diagram" width="100%"/>

---

<div align="center" style="color: #76B900; font-size: 1.4em; font-weight: bold;">
📊 Performance & Results
</div>

Based on the statistical validation of **317 real crossing events** under varied lighting and density conditions:

<style>
.table-green thead { background-color: #76B900; color: white; }
.table-green th, .table-green td { border: 1px solid #76B900; padding: 12px; text-align: center; }
</style>

<table class="table-green" width="100%">
  <thead>
    <tr>
      <th>Metric</th>
      <th>YOLOv5s + SORT (Optimized)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Precision</strong></td>
      <td>0.95</td>
    </tr>
    <tr>
      <td><strong>Recall</strong></td>
      <td>0.88</td>
    </tr>
    <tr>
      <td><strong>F1-Score</strong></td>
      <td>0.91</td>
    </tr>
    <tr>
      <td><strong>MOTA (Tracking Accuracy)</strong></td>
      <td>0.72</td>
    </tr>
    <tr>
      <td><strong>Inference Speed</strong></td>
      <td><span style="color: #76B900; font-size: 1.2em;">11 FPS</span><br><em>(Stable on Jetson Nano)</em></td>
    </tr>
  </tbody>
</table>

> **Finding:** YOLOv5s combined with the SORT algorithm proved to be the most stable and reliable configuration for this hardware compared to newer YOLOv11 variants.

---

<div align="center" style="color: #76B900; font-size: 1.4em; font-weight: bold;">
⚙️ Technologies Used
</div>

- **Hardware:** NVIDIA Jetson Nano (4GB), EasyULT Webcam Full HD 1080p  
- **Deep Learning:** YOLOv5s, YOLOv11s  
- **Inference Engine:** NVIDIA TensorRT (FP16), CUDA, cuDNN  
- **Tracking:** SORT (Simple Online and Realtime Tracking)  
- **Software:** Python, OpenCV, GStreamer, PyTorch

---

<div align="center" style="color: #76B900; font-size: 1.4em; font-weight: bold;">
🖥️ User Interface
</div>

The system features a graphical launcher that allows users to select models, tracking algorithms, and monitor live statistics.

<img src="images/interface.png" alt="User Interface" width="100%"/>

<em>Figure: Real-time interface showing detection bounding boxes, unique IDs, and IN/OUT counters.</em>

---

<div align="center">

### 👨‍💻 Author

**Andrés Fábregas**  
Electronic Engineer & Software Developer  

[![Portfolio](https://img.shields.io/badge/Portfolio-byandresfabregas-76B900?style=for-the-badge&logo=vercel)](https://byandresfabregas.vercel.app/)

</div>

<br>

<div align="center">
  <img src="https://img.shields.io/badge/Made_with_❤️-for_Edge_AI-brightgreen?style=flat" alt="Made with love"/>
</div>
