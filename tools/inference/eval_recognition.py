"""
tools/inference/eval_recognition.py
用途：识别准确率评估（CAR / 完全匹配率 / 每类准确率 / 混淆矩阵）
说明：
- 输入：真值 YOLO txt 目录 + 预测 YOLO txt 目录（class_id x y w h 格式）
- 输出：终端打印评估指标
- 运行方式：直接改下面 2 个路径，然后 python tools/eval_recognition.py
"""

from pathlib import Path
from collections import defaultdict

# ========== 改这里 ==========
GT_LABELS_DIR = Path("labels_gt")                                     # 改为实际真值 YOLO txt 目录路径
PRED_LABELS_DIR = Path("labels_pred")                                 # 改为实际预测 YOLO txt 目录路径
# ==========================

# 36 类 ID → 字符
CLASS_NAMES_36 = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]


def load_yolo_classes(txt_path):
    """YOLO 格式，提取 class_id 转字符"""
    if not txt_path.exists():
        return []
    chars = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id = int(line.split()[0])
            if 0 <= class_id < len(CLASS_NAMES_36):
                chars.append(CLASS_NAMES_36[class_id])
            else:
                chars.append(str(class_id))
    return chars


txt_files = sorted([p for p in GT_LABELS_DIR.glob("*.txt") if p.name != "classes.txt"])

total_chars = correct_chars = 0
exact_match_imgs = 0
total_imgs = 0

class_correct = defaultdict(int)
class_total = defaultdict(int)
confusion = defaultdict(lambda: defaultdict(int))

for txt_path in txt_files:
    pred_path = PRED_LABELS_DIR / txt_path.name
    if not pred_path.exists() or pred_path.name == "classes.txt":
        continue

    gt_chars = load_yolo_classes(txt_path)
    pred_chars = load_yolo_classes(PRED_LABELS_DIR / txt_path.name)

    if not gt_chars and not pred_chars:
        total_imgs += 1
        exact_match_imgs += 1
        continue
    if not gt_chars or not pred_chars:
        total_imgs += 1
        total_chars += max(len(gt_chars), len(pred_chars))
        for c in gt_chars:
            class_total[c] += 1
            confusion[c]["<MISSING>"] += 1
        continue

    total_imgs += 1
    min_len = min(len(gt_chars), len(pred_chars))

    img_correct = 0
    for i in range(min_len):
        gt = gt_chars[i]
        pred = pred_chars[i]
        class_total[gt] += 1
        confusion[gt][pred] += 1
        if gt == pred:
            correct_chars += 1
            class_correct[gt] += 1
            img_correct += 1

    total_chars += len(gt_chars)
    if img_correct == len(gt_chars) and len(gt_chars) == len(pred_chars):
        exact_match_imgs += 1

car = correct_chars / total_chars if total_chars else 0
exact_rate = exact_match_imgs / total_imgs if total_imgs else 0

print("\n" + "=" * 60)
print("  识 别 准 确 率")
print("=" * 60)
print(f"  总字符: {total_chars}  正确: {correct_chars}")
print(f"  CAR:            {car:.4f}  ({correct_chars}/{total_chars})")
print(f"  完全匹配率:     {exact_rate:.4f}  ({exact_match_imgs}/{total_imgs})")
print()
print("-" * 60)
print(f"  {'字符':<6} {'正确':<8} {'总数':<8} {'准确率':<8}")
print("-" * 60)
for cls in sorted(class_total.keys()):
    acc = class_correct[cls] / class_total[cls]
    print(f"  {cls:<6} {class_correct[cls]:<8} {class_total[cls]:<8} {acc:.4f}")
print("-" * 60)
print("\n  混淆矩阵（真值 → 预测，只显示错误）:")
print(f"  {'真值':<6} → {'预测':<6} 次数")
print("-" * 30)
for gt in sorted(confusion.keys()):
    for pred, cnt in sorted(confusion[gt].items(), key=lambda x: -x[1]):
        if gt != pred:
            print(f"  {gt:<6} → {pred:<6} {cnt}")
print("=" * 60)