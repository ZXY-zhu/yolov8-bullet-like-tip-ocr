"""
YOLOv8 训练脚本（CPU 调试用）
================================
说明：
- 本脚本仅用于本地 CPU 环境下的快速功能验证。
- 正式训练请在 GPU 环境下进行，并相应调整 epochs 与 batch 参数。
- 如需启用 GPU，将 device="cpu" 改为 device="0"。
"""
from ultralytics import YOLO

# 1. 加载预训练模型
model = YOLO("yolov8n.pt")

# 2. 开始训练
model.train(
    data="data.yaml",  # 本地数据配置文件（相对路径）
    epochs=5,
    imgsz=640,
    batch=4,
    device="cpu"
)
