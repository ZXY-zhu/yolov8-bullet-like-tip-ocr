"""
evaluate_accuracy.py
标准 OCR 评估脚本（CAR/CER + 位置准确率 + 混淆矩阵）

★ 修改：只评估 CSV 中存在的文件，避免全量标签导致的数据错位
"""
import os
import sys
import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ========== 路径配置 ==========
BASE_DIR = Path(__file__).parent.parent
LABELS_DIR = Path(r"C:\path\to\your\dataset\labels_raw")  # 改成你的 labels_raw 目录路径
CSV_PATH = Path(r"tools/outputs/recognition_results_v2_full.csv")  # 改成你刚跑的 CSV 路径

# ========== 类别映射（36 类：0-9 + A-Z）==========
CLASS_NAMES = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]
print(f"类别映射已加载，共 {len(CLASS_NAMES)} 类")

# ========== 加载 CSV 中已识别的文件名 ==========
if not CSV_PATH.exists():
    print(f"错误：找不到 {CSV_PATH}")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
recognized_files = set(df["filename"].tolist())
print(f"已加载 {len(recognized_files)} 条识别结果")

# ========== 加载标签（只加载 CSV 中存在的）==========
def parse_label_file(txt_path):
    """解析 YOLO 格式标签，提取字符序列"""
    chars = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                chars.append(CLASS_NAMES[class_id])
    return chars

label_files = list(LABELS_DIR.glob("*.txt"))
print(f"找到 {len(label_files)} 个标注文件")

# 只保留 CSV 中存在的
label_files = [p for p in label_files if p.stem + ".jpg" in recognized_files
               or p.stem + ".png" in recognized_files
               or p.stem + ".jpeg" in recognized_files]

print(f"实际参与评估的标注文件: {len(label_files)} 个")

if len(label_files) == 0:
    print("错误：没有匹配的标注文件，请检查 CSV 文件名和标签目录")
    sys.exit(1)

# ========== 构建 GT 字典 ==========
ground_truths = {}
for txt_path in label_files:
    # 尝试多种后缀匹配
    for ext in [".jpg", ".png", ".jpeg"]:
        img_name = txt_path.stem + ext
        if img_name in recognized_files:
            gt_chars = parse_label_file(txt_path)
            if len(gt_chars) > 4:
                gt_chars = gt_chars[:4]
                print(f"  ⚠ {txt_path.name}: 解析出 {len(gt_chars)} 个字符，截断为 4 个")
            ground_truths[img_name] = gt_chars
            break

print(f"\n总评估样本数: {len(ground_truths)}")
print(f"标签总字符数: {sum(len(v) for v in ground_truths.values())}")

# ========== 计算 CAR / CER ==========
total_chars = 0
correct_chars = 0
substitutions = 0
deletions = 0
insertions = 0

position_correct = [0, 0, 0, 0]
position_total = [0, 0, 0, 0]

confusion = defaultdict(int)

for _, row in df.iterrows():
    fname = row["filename"]
    pred_code = str(row["code"]) if pd.notna(row["code"]) else ""

    if fname not in ground_truths:
        continue

    gt_chars = ground_truths[fname]
    gt_code = "".join(gt_chars)

    total_chars += len(gt_chars)

    # 逐字符对比
    for i in range(min(len(gt_chars), len(pred_code))):
        gt_c = gt_chars[i]
        pred_c = pred_code[i]

        if gt_c == pred_c:
            correct_chars += 1
            if i < 4:
                position_correct[i] += 1
        else:
            substitutions += 1
            confusion[(i, gt_c, pred_c)] += 1

        if i < 4:
            position_total[i] += 1

    # 删除错误（GT 有但 Pred 没有）
    if len(pred_code) < len(gt_chars):
        deletions += len(gt_chars) - len(pred_code)

    # 插入错误（Pred 有但 GT 没有）
    if len(pred_code) > len(gt_chars):
        insertions += len(pred_code) - len(gt_chars)

# ========== 打印报告 ==========
print("\n" + "=" * 60)
print("准确率评估报告（标准 OCR 评估口径）")
print("=" * 60)

car = correct_chars / total_chars * 100 if total_chars > 0 else 0
cer = (substitutions + deletions + insertions) / total_chars * 100 if total_chars > 0 else 0

print(f"\n--- 核心指标 ---")
print(f"CAR (字符准确率): {correct_chars}/{total_chars} = {car:.2f}%")
print(f"CER (字符错误率): {substitutions + deletions + insertions}/{total_chars} = {cer:.2f}%")

print(f"\n各位置准确率（仅统计 0-3 位）:")
for i in range(4):
    if position_total[i] > 0:
        acc = position_correct[i] / position_total[i] * 100
        labels = ["首位", "第2位", "第3位", "第4位"]
        print(f"  {labels[i]}: {position_correct[i]}/{position_total[i]} = {acc:.2f}%")

# 4 位完全匹配
match_4 = sum(1 for _, row in df.iterrows()
              if row["filename"] in ground_truths
              and len(ground_truths[row["filename"]]) == 4
              and len(str(row["code"])) == 4
              and str(row["code"]) == "".join(ground_truths[row["filename"]]))
total_4 = sum(1 for _, row in df.iterrows()
              if row["filename"] in ground_truths
              and len(ground_truths[row["filename"]]) == 4)

print(f"\n4 位完全匹配率（仅 {total_4} 张 4 位标注图）: {match_4}/{total_4} = {match_4/total_4 * 100:.2f}%" if total_4 > 0 else "")

# 混淆详情
print(f"\n--- 混淆详情 (位置, 真实->识别, 次数) ---")
sorted_confusion = sorted(confusion.items(), key=lambda x: x[1], reverse=True)
for (pos, gt, pred), cnt in sorted_confusion[:30]:
    labels = ["首位", "第2位", "第3位", "第4位"]
    pos_label = labels[pos] if pos < 4 else f"第{pos+1}位"
    pred_label = pred if pred != "" else "(缺失)"
    print(f"  {pos_label}: {gt} -> {pred_label}  (出现 {cnt} 次)")

print("\n" + "=" * 60)