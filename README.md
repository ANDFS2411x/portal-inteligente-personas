<!-- ========================================================= -->
<!-- INTELLIGENT PEOPLE COUNTING PORTAL — FINAL SHOWCASE -->
<!-- ========================================================= -->

<div align="center">

<img
  src="https://capsule-render.vercel.app/api?type=rect&color=0:0d1117,100:0d1117&height=140&section=header&text=Intelligent%20People-Counting%20Portal&fontSize=40&fontColor=4EE31C&animation=fadeIn"
/>

<br/>

<img 
  src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=26&duration=2300&pause=600&color=4EE31C&center=true&vCenter=true&width=900&lines=Edge+AI+People+Counting+System;Real-Time+Computer+Vision+on+Jetson+Nano;Offline+Inference+Low+Latency+High+Privacy;Designed+for+Embedded+Vision" 
/>

<br/><br/>

<img src="https://img.shields.io/badge/🟢%20Edge%20AI-Jetson%20Nano-4EE31C?style=for-the-badge&logo=nvidia&logoColor=0d1117"/>
<img src="https://img.shields.io/badge/🟢%20TensorRT-FP16-66FF66?style=for-the-badge"/>
<img src="https://img.shields.io/badge/🟢%20YOLO-Optimized-2ECC71?style=for-the-badge"/>
<img src="https://img.shields.io/badge/✔%20Status-Research%20Project-4EE31C?style=for-the-badge"/>

<br/><br/>

<strong style="color:#7CFF9B;">
Computer Vision · Deep Learning · Edge Computing
</strong>

</div>

---

<div style="
background:#0d1117;
border-left:6px solid #4EE31C;
padding:18px;
color:#9AFF9A;
font-size:15px;
">

An <strong>Intelligent Portal System for People Counting</strong> using computer vision and deep learning, designed for <strong>Edge Computing</strong> on an <strong>NVIDIA Jetson Nano</strong>.<br/>
The system performs <strong>real-time, bidirectional counting</strong> (entries and exits) completely <strong>offline</strong>, ensuring <strong>low latency</strong> and <strong>data privacy</strong>.

</div>

---

## 🟢 Project Demo (Video)

<div align="center">

<a href="https://youtu.be/NFXqzJeZJb8?si=ROjr2mFycXRd_YXa" target="_blank">
  <img 
    src="https://img.youtube.com/vi/NFXqzJeZJb8/maxresdefault.jpg" 
    alt="Project Demo Video"
    width="85%"
    style="border:3px solid #4EE31C; border-radius:14px;"
  />
</a>

<br/><br/>

<strong style="color:#4EE31C;">
✔ Click to watch the full system demo on YouTube
</strong>

</div>

---

## 🟢 System Output (Live Inference)

<div align="center" style="
background:#0d1117;
border:2px solid #4EE31C;
border-radius:18px;
padding:22px;
margin-top:24px;
">

<img 
  src="images/video_001.gif" 
  alt="People Counting System Output"
  width="90%"
  style="border-radius:14px;"
/>

<br/><br/>

<strong style="color:#7CFF9B;">
✔ Real-time detection, tracking IDs, and IN / OUT counting running fully on-device
</strong>

</div>

---

## 🟢 Project Overview

<span style="color:#9AFF9A;">
This project implements an <strong>overhead (top-down) vision system</strong> to monitor occupant flow through access points.<br/>
By processing video frames locally, the system detects individuals, tracks their trajectories, and registers crossings through a virtual line.
</span>

<div style="
margin-top:18px;
padding:18px;
background:#0d1117;
border:1px solid #4EE31C55;
border-radius:14px;
color:#7CFF9B;
">

<strong>🟢 Core Capabilities</strong>

<table width="100%">
<tr>
<td width="50%">

✔ <strong>Real-Time Detection</strong><br/>
YOLO models optimized for embedded hardware<br/>
Low-latency inference<br/>
Overhead vision for robust tracking

</td>
<td width="50%">

✔ <strong>Privacy-First Design</strong><br/>
100% local processing<br/>
Zero cloud dependency<br/>
No external video transmission

</td>
</tr>
<tr>
<td width="50%">

✔ <strong>High Performance</strong><br/>
TensorRT FP16 acceleration<br/>
Multi-threaded execution<br/>
Stable 11 FPS operation

</td>
<td width="50%">

✔ <strong>Bidirectional Tracking</strong><br/>
Entry / exit counting<br/>
SORT-based trajectory analysis<br/>
Virtual line crossing logic

</td>
</tr>
</table>

</div>

---

## 🟢 System Architecture

<div align="center">

### General System Flow
![General System Diagram](images/system_diagram.png)

<br/>

### Internal Block Architecture
![Block Diagram](images/block_diagram.png)

</div>

---

## 🟢 Performance & Results

<div align="center">

<table style="width:80%;">
<thead>
<tr style="background:linear-gradient(90deg,#66FF6644,transparent); color:#66FF66;">
<th align="left">Metric</th>
<th align="center">YOLOv5s + SORT</th>
</tr>
</thead>
<tbody>
<tr><td>Precision</td><td align="center">0.95</td></tr>
<tr><td>Recall</td><td align="center">0.88</td></tr>
<tr><td>F1-Score</td><td align="center">0.91</td></tr>
<tr><td>MOTA</td><td align="center">0.72</td></tr>
<tr><td><strong>Inference Speed</strong></td><td align="center"><strong>11 FPS</strong></td></tr>
</tbody>
</table>

</div>

---

## 🟢 Tech Stack

<table width="100%">
<tr>
<td width="50%" valign="top">

### Hardware Platform
<pre style="background:#0d1117; color:#9AFF9A; border:1px solid #4EE31C55; border-radius:12px; padding:16px;">
NVIDIA Jetson Nano (4GB)
• 128 CUDA Cores
• Quad-core ARM Cortex-A57
• 4GB LPDDR4

EasyULT Webcam Full HD
• 1080p @ 30 FPS
• USB Interface
• Top-down mounting
</pre>

</td>
<td width="50%" valign="top">

### Software Stack
<pre style="background:#0d1117; color:#9AFF9A; border:1px solid #4EE31C55; border-radius:12px; padding:16px;">
YOLOv5s, YOLOv11s
TensorRT (FP16)
CUDA, cuDNN
OpenCV, GStreamer
SORT Tracking
PyTorch
</pre>

</td>
</tr>
</table>

---

<div align="center" style="
background:#0d1117;
border:2px solid #4EE31C;
border-radius:18px;
padding:30px;
margin-top:50px;
">

<img 
  src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=26&duration=2200&pause=900&color=4EE31C&center=true&vCenter=true&width=500&lines=Andr%C3%A9s+F%C3%A1bregas" 
/>

<br/>

<strong style="color:#9AFF9A; letter-spacing:1px;">
Electronic Engineer · Software Developer · Edge AI Specialist
</strong>

<br/><br/>

<a href="https://byandresfabregas.vercel.app/" target="_blank">
<img src="https://img.shields.io/badge/🟢%20Portfolio-byandresfabregas.vercel.app-4EE31C?style=for-the-badge"/>
</a>

<br/><br/>

<span style="color:#7CFF9B;">
✔ Building high-performance computer vision systems for the edge
</span>

</div>

---
