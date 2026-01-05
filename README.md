<div align="center">

# 🚪 Intelligent People-Counting Portal 📊

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Edge%20AI%20Vision%20System&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32" width="100%"/>

[![NVIDIA Jetson](https://img.shields.io/badge/NVIDIA-Jetson%20Nano%204GB-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/embedded/jetson-nano)
[![TensorRT](https://img.shields.io/badge/TensorRT-FP16%20Optimized-00FF00?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/tensorrt)
[![YOLOv5](https://img.shields.io/badge/YOLOv5s-Detection-FF6B6B?style=for-the-badge)](https://github.com/ultralytics/yolov5)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-10.2-76B900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

### 🎯 **Intelligent Portal System for People Counting** 
*Computer Vision + Deep Learning on Edge Computing*

**Real-time • Bidirectional • Privacy-First • 11 FPS on Embedded Hardware**

<br>

</div>

---

<div align="center">

## 📌 Project Overview

</div>

This project implements an **overhead (top-down) vision system** to monitor occupant flow through access points. By processing video frames locally, the system detects individuals, tracks their movement, and records crossings through a virtual line.

<br>

<div align="center">

### 🎯 Key Goals

</div>

<table>
<tr>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/96/000000/artificial-intelligence.png" width="80"/>

### 🔥 Real-time Detection
Leveraging YOLO models optimized for embedded hardware

</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/96/000000/microchip.png" width="80"/>

### 💻 Local Processing
100% offline operation on NVIDIA Jetson Nano

</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/96/000000/lightning-bolt.png" width="80"/>

### ⚡ High Performance
Optimized inference using NVIDIA TensorRT (FP16)

</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/96/000000/data-protection.png" width="80"/>

### 🔒 Data Privacy
No video data leaves the local device

</td>
</tr>
</table>

<br>

---

<div align="center">

## 🧠 System Architecture

</div>

<br>

### 1️⃣ General System Diagram

<div align="center">

*The complete flow from video acquisition and frame pre-processing to visualization*

<br>

![General System Diagram](images/system_diagram.png)

</div>

<br>

### 2️⃣ Block Diagram

<div align="center">

*Internal component interaction and data flow optimized for the Jetson Nano environment*

<br>

![Block Diagram](images/block_diagram.png)

</div>

<br>

---

<div align="center">

## 🔄 Optimization & Multithreading

</div>

To maintain a stable frame rate on low-resource hardware, the system utilizes a **three-thread architecture**:

<br>

<div align="center">

```mermaid
graph TD
    A[🎥 Capture Thread] -->|GStreamer Pipeline| B[Frame Buffer]
    B --> C[⚡ Processing Thread]
    C -->|TensorRT Inference| D[YOLO Detection]
    D -->|Multi-Object| E[SORT Tracking]
    E -->|Line Crossing| F[Count Logic]
    F --> G[📊 Visualization Thread]
    G -->|Real-time| H[🖥️ UI Display]
    
    style A fill:#76B900,stroke:#333,stroke-width:3px,color:#fff
    style C fill:#FF6B6B,stroke:#333,stroke-width:3px,color:#fff
    style G fill:#00D9FF,stroke:#333,stroke-width:3px,color:#fff
```

</div>

<br>

<table align="center">
<tr>
<td align="center" width="33%">

**🎥 Thread 1: Capture**  
Frame acquisition via GStreamer

</td>
<td align="center" width="33%">

**⚡ Thread 2: Processing**  
Inference (TensorRT), Tracking (SORT), counting logic

</td>
<td align="center" width="33%">

**📊 Thread 3: Visualization**  
Real-time UI rendering and statistics display

</td>
</tr>
</table>

<br>

### Model Conversion Pipeline (.pt → .onnx → .engine)

<div align="center">

*Deep learning models are converted to **TensorRT Engines with FP16 precision** to maximize CUDA core utilization*

<br>

```
╔═══════════════╗      ╔═══════════════╗      ╔═══════════════╗
║   YOLOv5s     ║      ║     ONNX      ║      ║   TensorRT    ║
║   .pt file    ║ ───► ║  Intermediate ║ ───► ║   .engine     ║
╚═══════════════╝      ╚═══════════════╝      ╚═══════════════╝
   PyTorch                  Universal           FP16 + CUDA
   Weights                  Format              Optimized
```

<br>

![Model Conversion Diagram](images/model_conversion.png)

</div>

<br>

---

<div align="center">

## 📊 Performance & Results

</div>

<div align="center">

### 🎯 Statistical Validation: **317 Real Crossing Events**
*Under varied lighting and density conditions*

<br>

<table>
<thead>
<tr>
<th align="left">🏆 Metric</th>
<th align="center">YOLOv5s + SORT (Optimized)</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><b>Precision</b></td>
<td align="center"><img src="https://img.shields.io/badge/0.95-95%25-success?style=for-the-badge" /></td>
</tr>
<tr>
<td align="left"><b>Recall</b></td>
<td align="center"><img src="https://img.shields.io/badge/0.88-88%25-success?style=for-the-badge" /></td>
</tr>
<tr>
<td align="left"><b>F1-Score</b></td>
<td align="center"><img src="https://img.shields.io/badge/0.91-91%25-success?style=for-the-badge" /></td>
</tr>
<tr>
<td align="left"><b>MOTA (Tracking Accuracy)</b></td>
<td align="center"><img src="https://img.shields.io/badge/0.72-72%25-yellow?style=for-the-badge" /></td>
</tr>
<tr>
<td align="left"><b>Inference Speed</b></td>
<td align="center"><img src="https://img.shields.io/badge/11_FPS-Stable-brightgreen?style=for-the-badge&logo=speedtest" /></td>
</tr>
</tbody>
</table>

</div>

<br>

> **💡 Finding:** YOLOv5s combined with the SORT algorithm proved to be the most stable and reliable configuration for this hardware compared to newer YOLOv11 variants.

<br>

---

<div align="center">

## ⚙️ Technologies Used

</div>

<br>

<div align="center">

<table>
<tr>
<td align="center" width="33%">

<img src="https://img.icons8.com/color/96/000000/nvidia.png" width="80"/>

### 🔧 Hardware

**NVIDIA Jetson Nano (4GB)**  
EasyULT Webcam Full HD 1080p

</td>
<td align="center" width="33%">

<img src="https://img.icons8.com/fluency/96/000000/brain.png" width="80"/>

### 🧠 Deep Learning

**YOLOv5s, YOLOv11s**  
NVIDIA TensorRT (FP16)  
CUDA, cuDNN

</td>
<td align="center" width="33%">

<img src="https://img.icons8.com/fluency/96/000000/code.png" width="80"/>

### 💻 Software

**Python, OpenCV**  
GStreamer, PyTorch  
SORT (Tracking)

</td>
</tr>
</table>

</div>

<br>

---

<div align="center">

## 🖥️ User Interface

</div>

<br>

<div align="center">

*The system features a graphical launcher that allows users to select models, tracking algorithms, and monitor live statistics*

<br>

![User Interface](images/interface.png)

<br>

**Real-time interface showing detection bounding boxes, unique IDs, and IN/OUT counters**

</div>

<br>

---

<div align="center">

## 👨‍💻 Author

<br>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=24&duration=3000&pause=1000&color=00D9FF&center=true&vCenter=true&width=435&lines=Andr%C3%A9s+F%C3%A1bregas" alt="Typing SVG" />

<br>

**Electronic Engineer & Software Developer**

<br>

[![Portfolio](https://img.shields.io/badge/🌐_Portfolio-Visit_Site-00D9FF?style=for-the-badge&logoColor=white)](https://byandresfabregas.vercel.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com)

</div>

<br>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer" width="100%"/>

**⚡ Powered by NVIDIA Jetson Nano | Built with ❤️ for Edge AI**

</div>
