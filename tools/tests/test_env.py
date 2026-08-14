"""
test_env.py
用途：验证 YOLOv8 运行环境
创建时间：2026-08-04
环境信息：
- Python 3.10.10
- torch 2.13.0+cpu
- ultralytics 8.4.115
- OpenCV 5.0.0
- CUDA: False (AMD Ryzen 7 7730U 无独显)
"""

import sys
print(sys.executable)

import torch
print(torch.__version__)
print(torch.cuda.is_available())

from ultralytics import YOLO
import cv2
print(cv2.__version__)