<div align="center">

# Intelligent People-Counting Portal 🚪📊

[![License: MIT](https://img.shields.io/badge/License-MIT-00FF7F?style=for-the-badge&logo=opensourceinitiative)](https://opensource.org/licenses/MIT)
[![Jetson Nano](https://img.shields.io/badge/Platform-NVIDIA_Jetson_Nano-00D26A?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/embedded/jetson-nano)
[![YOLO](https://img.shields.io/badge/Model-YOLOv5s-00FFFF?style=for-the-badge&logo=opencv)](https://ultralytics.com/yolo)
[![TensorRT](https://img.shields.io/badge/Optimized-TensorRT_FP16-00FF7F?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/tensorrt)
[![Edge AI](https://img.shields.io/badge/Edge_AI-100%25_Offline-00D26A?style=for-the-badge)](/)

**An Intelligent Portal System for People Counting** using computer vision and deep learning, designed for **Edge Computing** on an NVIDIA Jetson Nano.

Precise, real-time, bidirectional counting (entries and exits) — **no cloud, no latency, total privacy**.

</div>

---

<div align="center" style="color: #00FF7F; font-size: 1.5em; font-weight: bold;">
📌 Project Overview
</div>

Overhead (top-down) vision system that detects, tracks, and counts people crossing a virtual line — all processed **locally**.

### ✨ Key Goals
- <span style="color: #00FF7F;">✅</span> **Real-time Detection** — YOLO models optimized for embedded hardware  
- <span style="color: #00FF7F;">✅</span> **100% Offline** — Full processing on Jetson Nano  
- <span style="color: #00FF7F;">✅</span> **Max Performance** — TensorRT FP16 inference  
- <span style="color: #00FF7F;">✅</span> **Zero Data Leak** — Nothing leaves the device

---

<div align="center" style="color: #00FF7F; font-size: 1.5em; font-weight: bold;">
🧠 System Architecture
</div>

### 1️⃣ General System Diagram
<img src="images/system_diagram.png" alt="General System Diagram" width="100%"/>

### 2️⃣ Block Diagram
<img src="images/block_diagram.png" alt="Block Diagram" width="100%"/>

---

<div align="center" style="color: #00FF7F; font-size: 1.5em; font-weight: bold;">
🔄 Optimization & Multithreading
</div>

Three-thread architecture for smooth performance on limited hardware:

1. **Capture Thread** → GStreamer frame acquisition  
2. **Processing Thread** → TensorRT inference + SORT tracking + counting  
3. **Visualization Thread** → Live UI and stats

### Model Conversion Pipeline
.pt → .onnx → **.engine (FP16)**

<img src="images/model_conversion.png" alt="Model Conversion" width="100%"/>

---

<div align="center" style="color: #00FF7F; font-size: 1.5em; font-weight: bold;">
📊 Performance & Results
</div>

Validated on **317 real crossing events** (varying light & density):

<table width="100%" style="border-collapse: collapse; margin: 20px 0;">
  <thead>
    <tr style="background-color: #00D26A; color: white;">
      <th style="padding: 12px; border: 2px solid #00FF7F;">Metric</th>
      <th style="padding: 12px; border: 2px solid #00FF7F;">YOLOv5s + SORT (Optimized)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 10px; border: 1px solid #00FF7F;"><strong>Precision</strong></td>
      <td style="padding: 10px; border: 1px solid #00FF7F; text-align: center;">0.95</td>
    </tr>
    <tr style="background-color: #001a0f;">
      <td style="padding: 10px; border: 1px solid #00FF7F;"><strong>Recall</strong></td>
      <td style="padding: 10px; border: 1px solid #00FF7F; text-align: center;">0.88</td>
    </tr>
    <tr>
      <td style="padding: 10px; border: 1px solid #00FF7F;"><strong>F1-Score</strong></td>
      <td style="padding: 10px; border: 1px solid #00FF7F; text-align: center;">0.91</td>
    </tr>
    <tr style="background-color: #001a0f;">
      <td style="padding: 10px; border: 1px solid #00FF7F;"><strong>MOTA</strong></td>
      <td style="padding: 10px; border: 1px solid #00FF7F; text-align: center;">0.72</td>
    </tr>
    <tr>
      <td style="padding: 10px; border: 1px solid #00FF7F;"><strong>Inference Speed</strong></td>
      <td style="padding: 10px; border: 1px solid #00FF7F; text-align: center;"><span style="color: #00FF7F; font-size: 1.3em; font-weight: bold;">11 FPS</span><br><em>Stable on Jetson Nano</em></td>
    </tr>
  </tbody>
</table>

> **Finding:** YOLOv5s + SORT was the most stable and reliable combo for this hardware.

---

<div align="center" style="color: #00FF7F; font-size: 1.5em; font-weight: bold;">
⚙️ Technologies Used
</div>

- **Hardware:** NVIDIA Jetson Nano (4GB), EasyULT Webcam 1080p  
- **Deep Learning:** YOLOv5s, YOLOv11s  
- **Inference:** NVIDIA TensorRT (FP16), CUDA, cuDNN  
- **Tracking:** SORT  
- **Software:** Python, OpenCV, GStreamer, PyTorch

---

<div align="center" style="color: #00FF7F; font-size: 1.5em; font-weight: bold;">
🖥️ User Interface
</div>

Graphical launcher with model selection, live stats, and real-time visualization.

<img src="images/interface.png" alt="User Interface" width="100%"/>

<em>Real-time view: bounding boxes, unique IDs, IN/OUT counters</em>

---

<div align="center">

### 👨‍💻 Author

![Visitor Badge](https://img.shields.io/badge/Visitor-Count-00FF7F?style=for-the-badge)
![Author](https://img.shields.io/badge/Author-Andr%C3%A9s_F%C3%A1bregas-00FF7F?style=for-the-badge&logo=github)

**Andrés Fábregas**  
<span style="color: #00FF7F; font-size: 1.1em;">Electronic Engineer & Software Developer</span>

[![Portfolio](https://img.shields.io/badge/Portfolio-Visit_Now-00D26A?style=for-the-badge&logo=vercel&logoColor=white)](https://byandresfabregas.vercel.app/)

</div>

<br>

<div align="center">
  <img src="https://img.shields.io/badge/Built_with_⚡_Edge_AI-00FF7F?style=flat-square" alt="Edge AI"/>
  <img src="https://img.shields.io/badge/Privacy-First-00FF7F?style=flat-square" alt="Privacy"/>
</div>
