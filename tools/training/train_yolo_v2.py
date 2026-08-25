"""
tools/training/train_yolo_v2.py
用途：YOLOv8 检测模型微调训练（v2，AutoDL 环境）
说明：
- 自动将 images/ + labels/ 按 8:2 拆分为 train/ val/ 子目录（仅首次运行）
- 基于已有 best.pt 权重继续微调
- 运行方式：直接改下面 root 和 model/train 中的路径，然后 python tools/training/train_yolo_v2.py
"""

import os
import random
import shutil
from pathlib import Path
from ultralytics import YOLO

random.seed(42)

# ========== 改这里 ==========
ROOT = Path("data")                                        # 改为实际数据集根目录路径
BEST_WEIGHTS = "weights/best.pt"                            # 改为实际预训练权重路径
DATA_YAML = "data.yaml"                                    # 改为实际 data.yaml 配置文件路径
PROJECT_DIR = "runs/detect"                                # 改为实际训练输出目录路径
# ==========================

img_dir = ROOT / "images"
lbl_dir = ROOT / "labels"

# 如果还没拆过（没有 train/ 子目录），就先拆
if not (img_dir / "train").exists():
    print("📂 还没拆 train/val，先拆...")
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    random.shuffle(img_files)
    val_count = int(len(img_files) * 0.2)

    for split, files in [("train", img_files[val_count:]), ("val", img_files[:val_count])]:
        (img_dir / split).mkdir(parents=True, exist_ok=True)
        (lbl_dir / split).mkdir(parents=True, exist_ok=True)
        for img_path in files:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            shutil.move(str(img_path), str(img_dir / split / img_path.name))
            if lbl_path.exists():
                shutil.move(str(lbl_path), str(lbl_dir / split / lbl_path.name))

    print(f"✅ 拆完: train {len(img_files[val_count:])} 张, val {val_count} 张")

# 然后训练
model = YOLO(BEST_WEIGHTS)
model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=32,
    lr0=1e-4,
    lrf=0.01,
    cos_lr=True,
    patience=15,
    device="0",
    workers=4,
    project=PROJECT_DIR,
    name="bullet_tip_v2",
    exist_ok=True,
)
print("✅ 训练完成！")