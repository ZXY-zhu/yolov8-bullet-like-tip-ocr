"""
tools/training/train.py
用途：YOLOv8 训练脚本（CPU 调试用）
说明：
- 本脚本仅用于本地 CPU 环境下的快速功能验证
- 正式训练请在 GPU 环境下进行，并相应调整 epochs 与 batch 参数
- 如需启用 GPU，将 device="cpu" 改为 device="0"
- 运行方式：直接改下面 data 路径，然后 python tools/training/train.py
"""

from ultralytics import YOLO

# ========== 改这里 ==========
DATA_YAML = "data.yaml"   # 改为实际 data.yaml 配置文件路径
# ==========================

# 1. 加载预训练模型
model = YOLO("yolov8n.pt")

# 2. 开始训练
model.train(
    data=DATA_YAML,
    epochs=5,
    imgsz=640,
    batch=4,
    device="cpu"
)