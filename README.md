# REAL-TIME-OBJECT-DETECTION

This project demonstrates a real-time augmented reality (AR) system combining object detection and semantic segmentation using deep learning. It uses **YOLOv5s** for object detection and **DeepLabV3 with MobileNetV3** for semantic segmentation. The system runs in real time on CPU and overlays bounding boxes and segmentation masks directly onto the webcam feed.

## 🔍 Features

- 🎯 Real-time object detection using YOLOv5 (COCO pretrained)
- 🧠 Semantic segmentation using DeepLabV3 with MobileNetV3 backbone
- 🧩 Combined detection + segmentation overlay
- 🧵 Multithreaded inference using Python’s `ThreadPoolExecutor`
- ⚡ Runs on CPU with 6–10 FPS (no GPU required)
