<div align="center">

# <span style="color:#76B900; text-shadow: 0 0 10px #76B900, 0 0 20px #76B900;">Intelligent People-Counting Portal</span> 🚪📊

[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](https://opensource.org/licenses/MIT)
[![Jetson Nano](https://img.shields.io/badge/Platform-NVIDIA_Jetson_Nano-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/embedded/jetson-nano)
[![YOLO](https://img.shields.io/badge/Model-YOLOv5s-00FFFF?style=for-the-badge&logo=opencv&logoColor=black)](https://ultralytics.com/yolo)
[![TensorRT](https://img.shields.io/badge/Optimized-TensorRT_FP16-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)

<span style="font-size:1.1em; color:#A0D911;">
**An Intelligent Portal System for People Counting** using computer vision and deep learning, designed for **Edge Computing** on an NVIDIA Jetson Nano.
</span>

<span style="color:#76B900;">
This system enables **precise, real-time, bidirectional counting** (entries and exits) **without cloud dependency**, ensuring low latency and complete data privacy.
</span>

<br>

[![Edge AI](https://img.shields.io/badge/Edge_AI-100%25_Offline-76B900?style=flat-square&logo=raspberrypi&logoColor=white)](https://developer.nvidia.com/embedded)
[![Privacy](https://img.shields.io/badge/Data_Privacy-No_Cloud-brightgreen?style=flat-square&logo=shield&logoColor=white)](https://developer.nvidia.com/embedded)

</div>

---

<div align="center" style="color: #76B900; font-size: 1.5em; font-weight: bold; text-shadow: 0 0 8px #76B900;">
📌 Project Overview
</div>

This project implements an **overhead (top-down) vision system** to monitor occupant flow through access points. By processing video frames locally, the system detects individuals, tracks their movement, and records crossings through a virtual line.

### ✨ Key Goals
- <span style="color: #76B900;">✅</span> **Real-time Detection:** Leveraging YOLO models optimized for embedded hardware  
- <span style="color: #76B900;">✅</span> **Local Processing:** 100% offline operation on NVIDIA Jetson Nano  
- <span style="color: #76B900;">✅</span> **High Performance:** Optimized inference using NVIDIA TensorRT (FP16)  
- <span style="color: #76B900;">✅</span> **Data Privacy:** No video data leaves the local device

---

<div align="center" style="color: #76B900; font-size: 1.5em; font-weight: bold; text-shadow: 0 0 8px #76B900;">
🧠 System Architecture
</div>

### 1️⃣ General System Diagram
<img src="images/system_diagram.png" alt="General System Diagram" width="100%"/>

### 2️⃣ Block Diagram
<img src="images/block_diagram.png" alt="Block Diagram" width="100%"/>

---

<div align="center" style="color: #76B900; font-size: 1.5em; font-weight: bold; text-shadow: 0 0 8px #76B900;">
🔄 Optimization & Multithreading
</div>

To maintain a stable frame rate on low-resource hardware, the system utilizes a **three-thread architecture**:

1. **Capture Thread** — Frame acquisition via GStreamer  
2. **Processing Thread** — Inference (TensorRT), Tracking (SORT), and counting logic  
3. **Visualization Thread** — Real-time UI rendering and statistics display

### Model Conversion Pipeline (.pt → .onnx → .engine)
<img src="images/model_conversion.png" alt="Model Conversion Diagram" width="100%"/>

---

<div align="center" style="color: #76B900; font-size: 1.5em; font-weight: bold; text-shadow: 0 0 8px #76B900;">
📊 Performance & Results
</div>

Based on the statistical validation of **317 real crossing events** under varied lighting and density conditions:

<table width="100%" style="border-collapse: collapse; margin: 20px 0;">
  <thead>
    <tr style="background-color: #76B900; color: white;">
      <th style="padding: 12px; border: 1px solid #76B900;">Metric</th>
      <th style="padding: 12px; border: 1px solid #76B900;">YOLOv5s + SORT (Optimized)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 12px; border: 1px solid #76B900;"><strong>Precision</strong></td>
      <td style="padding: 12px; border: 1px solid #76B900; text-align: center;">0.95</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #76B900;"><strong>Recall</strong></td>
      <td style="padding: 12px; border: 1px solid #76B900; text-align: center;">0.88</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #76B900;"><strong>F1-Score</strong></td>
      <td style="padding: 12px; border: 1px solid #76B900; text-align: center;">0.91</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #76B900;"><strong>MOTA (Tracking Accuracy)</strong></td>
      <td style="padding: 12px; border: 1px solid #76B900; text-align: center;">0.72</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #76B900;"><strong>Inference Speed</strong></td>
      <td style="padding: 12px; border: 1px solid #76B900; text-align: center;"><span style="color: #76B900; font-size: 1.3em; font-weight: bold;">11 FPS</span><br><em>(Stable on Jetson Nano)</em></td>
    </tr>
  </tbody>
</table>

> **Finding:** YOLOv5s combined with the SORT algorithm proved to be the most stable and reliable configuration for this hardware compared to newer YOLOv11 variants.

---

<div align="center" style="color: #76B900; font-size: 1.5em; font-weight: bold; text-shadow: 0 0 8px #76B900;">
⚙️ Technologies Used
</div>

- **Hardware:** NVIDIA Jetson Nano (4GB), EasyULT Webcam Full HD 1080p  
- **Deep Learning:** YOLOv5s, YOLOv11s  
- **Inference Engine:** NVIDIA TensorRT (FP16), CUDA, cuDNN  
- **Tracking:** SORT (Simple Online and Realtime Tracking)  
- **Software:** Python, OpenCV, GStreamer, PyTorch

---

<div align="center" style="color: #76B900; font-size: 1.5em; font-weight: bold; text-shadow: 0 0 8px #76B900;">
🖥️ User Interface
</div>

<img src="images/interface.png" alt="User Interface" width="100%"/>

<em>Figure: Real-time interface showing detection bounding boxes, unique IDs, and IN/OUT counters.</em>

---

<div align="center">

### <span style="color:#76B900; font-size:2em; font-weight:bold; text-shadow: 0 0 15px #76B900, 0 0 30px #76B900; animation: pulse 2s infinite;">👨‍💻 Andrés Fábregas</span>

<span style="color:#A0D911;">Electronic Engineer & Software Developer</span>

<br>

[![Portfolio](https://img.shields.io/badge/Portfolio-byandresfabregas.vercel.app-76B900?style=for-the-badge&logo=vercel&logoColor=white)](https://byandresfabregas.vercel.app/)

<br><br>

<span style="font-size:0.9em; color:#76B900;">
Made with ❤️ and a lot of green vibes for Edge AI
</span>

</div>

<style>
@keyframes pulse {
  0% { text-shadow: 0 0 15px #76B900, 0 0 30px #76B900; }
  50% { text-shadow: 0 0 20px #76B900, 0 0 40px #76B900, 0 0 50px #76B900; }
  100% { text-shadow: 0 0 15px #76B900, 0 0 30px #76B900; }
}
</style>
