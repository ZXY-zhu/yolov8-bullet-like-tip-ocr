"""
tools/inference/eval_detection.py
用途：检测准确率评估（基于 IoU 匹配计算 Recall / Precision / F1 / 平均 IoU）
说明：
- 输入：待检测图片目录、模型预测框目录、人工标注真值框目录
- 输出：终端打印评估指标
- 运行方式：直接改下面 3 个路径，然后 python tools/eval_detection.py
"""

from pathlib import Path
from PIL import Image as PILImage
import numpy as np

# ========== 改这里 ==========
IMAGES_DIR = Path("images")                                    # 改为实际图片目录路径
PRED_DIR = Path("labels_pred")                                 # 改为实际模型预测框目录路径
GT_DIR = Path("labels_gt")                                     # 改为实际人工标注真值框目录路径
IOU_THRESH = 0.5
# ============================


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def load_boxes(txt_path, img_w, img_h):
    boxes = []
    if not txt_path.exists():
        return boxes
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, x_c, y_c, bw, bh = parts[:5]
            x_c, y_c, bw, bh = map(float, [x_c, y_c, bw, bh])
            x1 = (x_c - bw / 2) * img_w
            y1 = (y_c - bh / 2) * img_h
            x2 = (x_c + bw / 2) * img_w
            y2 = (y_c + bh / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
img_paths = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in img_exts])

total_gt = total_pred = matched_gt = matched_pred = 0
ious = []

for img_path in img_paths:
    with PILImage.open(img_path) as im:
        img_w, img_h = im.size

    gt_boxes = load_boxes(GT_DIR / f"{img_path.stem}.txt", img_w, img_h)
    pred_boxes = load_boxes(PRED_DIR / f"{img_path.stem}.txt", img_w, img_h)

    total_gt += len(gt_boxes)
    total_pred += len(pred_boxes)

    matched = set()
    for pb in pred_boxes:
        best_iou, best_idx = 0, -1
        for i, gb in enumerate(gt_boxes):
            if i in matched:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= IOU_THRESH and best_idx >= 0:
            matched.add(best_idx)
            matched_pred += 1
            ious.append(best_iou)

    matched_gt += len(matched)

recall = matched_gt / total_gt if total_gt else 0
precision = matched_pred / total_pred if total_pred else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
mean_iou = np.mean(ious) if ious else 0

print("\n" + "=" * 50)
print("  检 测 准 确 率")
print("=" * 50)
print(f"  真值框: {total_gt}  预测框: {total_pred}")
print(f"  Recall:    {recall:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  F1:        {f1:.4f}")
print(f"  平均 IoU:  {mean_iou:.4f}")
print("=" * 50)