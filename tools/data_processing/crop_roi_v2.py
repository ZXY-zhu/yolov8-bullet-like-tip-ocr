"""
tools/data_processing/crop_roi_v2.py
用途：从 YOLO 格式标注的工业刻印数据集中，按检测框裁剪 ROI 并分类保存到对应类别子目录
说明：
- 输入：YOLO txt 标注 + 原始图片（支持 .jpg / .png）
- 输出：按类别名命名的文件夹，每个文件夹内存放对应字符的 ROI 裁剪图（RGB 原始色彩）
- 类别映射：自动从 labels_raw/classes.txt 读取，若不存在则默认 0-9 + A-Z（36 类）
- 依赖：opencv-python, numpy
- 运行环境：Python 3.10+
"""

import cv2
import numpy as np
import os
import shutil
from pathlib import Path

# ========== 1. 路径配置 ==========
BASE = Path("data/new_old_mix")   # 改为实际数据集根目录路径
IMG_DIR = BASE / "images"
LBL_DIR = BASE / "labels"
SAVE_ROOT = BASE / "dataset_cnn_rgb"   # 改为实际输出目录路径

# ========== 2. 类别映射 ==========
classes_file = BASE / "labels_raw" / "classes.txt"
if classes_file.exists():
    with open(classes_file) as f:
        CLASS_NAMES = [l.strip() for l in f.readlines() if l.strip()]
else:
    CLASS_NAMES = [str(i) for i in range(10)] + [chr(65 + i) for i in range(26)]

KEEP_CLASSES = {i: c for i, c in enumerate(CLASS_NAMES)}
print(f"📋 共 {len(KEEP_CLASSES)} 类: {CLASS_NAMES}")

# ========== 3. 创建输出目录（清空旧数据） ==========
if SAVE_ROOT.exists():
    shutil.rmtree(SAVE_ROOT)
for name in KEEP_CLASSES.values():
    (SAVE_ROOT / name).mkdir(parents=True, exist_ok=True)

# ========== 4. 逐图裁剪 ROI ==========
counts = {}

for txt_name in sorted(os.listdir(LBL_DIR)):
    if not txt_name.endswith(".txt") or txt_name == "classes.txt":
        continue

    txt_path = LBL_DIR / txt_name

    img_name = txt_name.replace(".txt", ".jpg")
    img_path = IMG_DIR / img_name
    if not img_path.exists():
        img_name = txt_name.replace(".txt", ".png")
        img_path = IMG_DIR / img_name
        if not img_path.exists():
            continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]

    with open(txt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            if class_id not in KEEP_CLASSES:
                continue

            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = max(0, int((xc - bw / 2) * w))
            y1 = max(0, int((yc - bh / 2) * h))
            x2 = min(w, int((xc + bw / 2) * w))
            y2 = min(h, int((yc + bh / 2) * h))

            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            char_name = KEEP_CLASSES[class_id]
            char_dir = SAVE_ROOT / char_name
            save_name = f"{Path(img_name).stem}_{class_id}_{counts.get(char_name, 0):04d}.png"
            cv2.imwrite(str(char_dir / save_name), roi)
            counts[char_name] = counts.get(char_name, 0) + 1

    print(f"  ✅ {img_name}")

# ========== 5. 统计输出 ==========
print("\n🎉 裁剪完成！每类数量：")
for idx, name in sorted(KEEP_CLASSES.items()):
    print(f"  {name} (id={idx}): {counts.get(name, 0)} 张")
print(f"总计: {sum(counts.values())} 张")