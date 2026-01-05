<div align="center">

# Intelligent People-Counting Portal 🚪📊

[![NVIDIA Jetson](https://img.shields.io/badge/NVIDIA-Jetson%20Nano-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/embedded/jetson-nano)
[![TensorRT](https://img.shields.io/badge/TensorRT-FP16-76B900?style=for-the-badge)](https://developer.nvidia.com/tensorrt)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

An **Intelligent Portal System for People Counting** using computer vision and deep learning, designed for **Edge Computing** on an NVIDIA Jetson Nano. This system enables precise, real-time, bidirectional counting (entries and exits) without cloud dependency, ensuring low latency and data privacy.

</div>

---

## 📌 Project Overview

This project implements an overhead (top-down) vision system to monitor occupant flow through access points. By processing video frames locally, the system detects individuals, tracks their movement, and records crossings through a virtual line.

### Key Goals

<table>
<tr>
<td width="50%">

**🔥 Real-time Detection**  
Leveraging YOLO models optimized for embedded hardware.

**💻 Local Processing**  
100% offline operation on NVIDIA Jetson Nano.

</td>
<td width="50%">

**⚡ High Performance**  
Optimized inference using NVIDIA TensorRT (FP16).

**🔒 Data Privacy**  
No video data leaves the local device.

</td>
</tr>
</table>

---

## 🧠 System Architecture

<details open>
<summary><h3>1️⃣ General System Diagram</h3></summary>

The complete flow from video acquisition and frame pre-processing to visualization.

![General System Diagram](images/system_diagram.png)

</details>

<details open>
<summary><h3>2️⃣ Block Diagram</h3></summary>

Internal component interaction and data flow optimized for the Jetson Nano environment.

![Block Diagram](images/block_diagram.png)

</details>

---

## 🔄 Optimization & Multithreading

To maintain a stable frame rate on low-resource hardware, the system utilizes a **three-thread architecture**:

```
┌─────────────────────┐
│  Capture Thread     │ ──► Frame acquisition via GStreamer
└─────────────────────┘

┌─────────────────────┐
│ Processing Thread   │ ──► Inference (TensorRT), Tracking (SORT), counting logic
└─────────────────────┘

┌─────────────────────┐
│Visualization Thread │ ──► Real-time UI rendering and statistics display
└─────────────────────┘
```

### Model Conversion Pipeline (.pt → .onnx → .engine)

Deep learning models are converted to **TensorRT Engines with FP16 precision** to maximize CUDA core utilization.

<div align="center">

![Model Conversion Diagram](images/model_conversion.png)

</div>

---

## 📊 Performance & Results

Based on the statistical validation of **317 real crossing events** under varied lighting and density conditions:

<div align="center">

| Metric | YOLOv5s + SORT (Optimized) |
|:---|:---:|
| **Precision** | `0.95` |
| **Recall** | `0.88` |
| **F1-Score** | `0.91` |
| **MOTA (Tracking Accuracy)** | `0.72` |
| **Inference Speed** | **`11 FPS`** (Stable on Jetson Nano) |

</div>

> **💡 Finding:** YOLOv5s combined with the SORT algorithm proved to be the most stable and reliable configuration for this hardware compared to newer YOLOv11 variants.

---

## ⚙️ Technologies Used

<table>
<tr>
<td align="center" width="33%">

### 🔧 Hardware
NVIDIA Jetson Nano (4GB)  
EasyULT Webcam Full HD 1080p

</td>
<td align="center" width="33%">

### 🧠 Deep Learning
YOLOv5s, YOLOv11s  
TensorRT (FP16), CUDA, cuDNN

</td>
<td align="center" width="33%">

### 💻 Software
Python, OpenCV  
GStreamer, PyTorch  
SORT (Tracking)

</td>
</tr>
</table>

---

## 🖥️ User Interface

The system features a graphical launcher that allows users to select models, tracking algorithms, and monitor live statistics.

<div align="center">

![User Interface](images/interface.png)

*Real-time interface showing detection bounding boxes, unique IDs, and IN/OUT counters.*

</div>

---

<div align="center">

## 👨‍💻 Author

**Andrés Fábregas**  
*Electronic Engineer & Software Developer*

[![Portfolio](https://img.shields.io/badge/Portfolio-Visit%20Site-00D9FF?style=for-the-badge&logo=vercel&logoColor=white)](https://byandresfabregas.vercel.app/)

</div>

---

<div align="center">

**Made with ❤️ and NVIDIA Jetson Nano**

</div>
