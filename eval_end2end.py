"""
eval_end2end.py
用途：端到端识别准确率评估（按 y 中心从上到下排序后逐位比对）
说明：
- GT 和预测均为 YOLO txt 格式（class_id x y w h），刻印为竖排
- 按 y_center 排序后逐字符比对，输出 CAR（仅字符）、CAR（含0框匹配）、完全匹配率
- 同时输出每类准确率及混淆矩阵（仅错误项）
- 运行方式：直接改下面 2 个路径，然后 python eval_end2end.py
"""

from pathlib import Path
from collections import defaultdict

# ========== 改这里 ==========
GT_LABELS_DIR = Path("labels_raw")                         # 改为实际 GT 标注目录路径
PRED_LABELS_DIR = Path("labels_pred")                      # 改为实际预测标注目录路径
# ==========================

CLASS_NAMES_36 = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]


def load_yolo_entries(txt_path):
    if not txt_path.exists():
        return []
    entries = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = int(parts[0])
            y_center = float(parts[2])
            char = CLASS_NAMES_36[class_id] if 0 <= class_id < len(CLASS_NAMES_36) else str(class_id)
            entries.append((char, y_center))
    return entries


txt_files = sorted([p for p in GT_LABELS_DIR.glob("*.txt") if p.name != "classes.txt"])

total_chars = correct_chars = 0
exact_match_imgs = 0
total_imgs = 0
zero_match_imgs = 0  # ★ 0框对0框计数

class_correct = defaultdict(int)
class_total = defaultdict(int)
confusion = defaultdict(lambda: defaultdict(int))

for txt_path in txt_files:
    pred_path = PRED_LABELS_DIR / txt_path.name
    if not pred_path.exists() or pred_path.name == "classes.txt":
        continue

    gt_entries = load_yolo_entries(txt_path)
    pred_entries = load_yolo_entries(pred_path)

    gt_entries.sort(key=lambda e: e[1])
    pred_entries.sort(key=lambda e: e[1])

    gt_chars = [e[0] for e in gt_entries]
    pred_chars = [e[0] for e in pred_entries]

    total_imgs += 1

    # ★ 0框 vs 0框 → 两个指标都算
    if not gt_chars and not pred_chars:
        zero_match_imgs += 1
        exact_match_imgs += 1
        continue

    if not gt_chars or not pred_chars:
        total_chars += max(len(gt_chars), len(pred_chars))
        for c in gt_chars:
            class_total[c] += 1
            confusion[c]["<MISSING>"] += 1
        continue

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

# ========== 汇总 ==========
car = correct_chars / total_chars if total_chars else 0
car_with_zero = (correct_chars + zero_match_imgs) / (total_chars + zero_match_imgs) if (total_chars + zero_match_imgs) else 0
exact_rate = exact_match_imgs / total_imgs if total_imgs else 0

print("\n" + "=" * 60)
print("  端到端识别准确率（按 y 排序）")
print("=" * 60)
print(f"  总图片: {total_imgs}")
print(f"  0框匹配图片: {zero_match_imgs}")
print(f"  总字符: {total_chars}  正确: {correct_chars}")
print(f"  CAR（仅字符）:      {car:.4f}  ({correct_chars}/{total_chars})")
print(f"  CAR（含0框匹配）:   {car_with_zero:.4f}  ({correct_chars + zero_match_imgs}/{total_chars + zero_match_imgs})")
print(f"  完全匹配率:         {exact_rate:.4f}  ({exact_match_imgs}/{total_imgs})")
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