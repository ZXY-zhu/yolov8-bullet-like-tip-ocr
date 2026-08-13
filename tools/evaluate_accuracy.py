"""
evaluate_accuracy.py
修正版评估：标签几位比几位，采用标准 OCR 指标
- CAR (Character Accuracy Rate): 正确字符数 / 标签总字符数
- CER (Character Error Rate): 基于 Levenshtein 编辑距离
- 位置级准确率（仅统计 0-3 位）
- 混淆矩阵

参照工业 OCR 评估标准：字符准确率 + 编辑距离
"""
import pandas as pd
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(r"C:\Users\BeLig\PycharmProjects\yolov8-bullet-like-tip-ocr")
LABELS_DIR = Path(r"C:\Users\BeLig\Desktop\dataset\labels_raw")
# ★ 改成你二进程的输出文件
RESULTS_CSV = BASE_DIR / "recognition_results_mp2.csv"

# ========== 1. 类别映射 ==========
classes_file = LABELS_DIR / "classes.txt"
with open(classes_file, 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f if line.strip()]
CLASS_MAP = {i: name for i, name in enumerate(class_names)}
print(f"类别映射已加载，共 {len(CLASS_MAP)} 类\n")


# ========== 2. 解析标注（变长，容错处理）==========
def parse_label_variable_len(txt_path: Path):
    """
    解析标注文件，返回按 y 从下到上排序的字符列表
    容错：如果解析出超过 4 个字符，只取前 4 个（按位置）
    """
    boxes = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            cy = float(parts[2])
            char = CLASS_MAP.get(class_id, '?')
            boxes.append((cy, char))

    # 按 y 从大到小排序（从下到上）
    boxes.sort(key=lambda b: b[0], reverse=True)
    chars = [b[1] for b in boxes]

    # 容错：如果超过 4 个，只保留前 4 个（避免越界）
    if len(chars) > 4:
        print(f"  ⚠ {txt_path.name}: 解析出 {len(chars)} 个字符，截断为 4 个")
        chars = chars[:4]

    return chars


# ========== 3. Levenshtein 编辑距离 ==========
def levenshtein(a: str, b: str) -> int:
    """计算两个字符串的编辑距离"""
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # 删除
                dp[i][j - 1] + 1,  # 插入
                dp[i - 1][j - 1] + cost  # 替换
            )
    return dp[m][n]


# ========== 4. 读取识别结果 ==========
if not RESULTS_CSV.exists():
    print(f"错误: 找不到 {RESULTS_CSV}")
    print("请先运行 predict_batch_mp2.py 生成识别结果")
    exit(1)

df = pd.read_csv(RESULTS_CSV)
results_dict = dict(zip(df['filename'], df['code']))
print(f"已加载 {len(results_dict)} 条识别结果\n")

# ========== 5. 逐张评估 ==========
total_gt_chars = 0  # 标签总字符数
total_correct_chars = 0  # 正确识别的字符数
total_edit_distance = 0  # 总编辑距离
total_samples = 0  # 评估样本数

exact_match_4 = 0  # 4 位完全匹配数
images_with_4_gt = 0  # 标签 4 位的图片数

# 位置级准确率（仅 0-3 位）
pos_correct = [0, 0, 0, 0]
pos_total = [0, 0, 0, 0]

confusion = []
errors = []

txt_files = sorted(LABELS_DIR.glob("*.txt"))
print(f"找到 {len(txt_files)} 个标注文件\n")

for txt_path in txt_files:
    gt_chars = parse_label_variable_len(txt_path)
    if not gt_chars:
        continue

    total_samples += 1
    gt_len = len(gt_chars)
    gt_code = ''.join(gt_chars)
    total_gt_chars += gt_len

    if gt_len == 4:
        images_with_4_gt += 1

    img_name = txt_path.stem + ".jpg"
    pred_code = results_dict.get(img_name, "") or ""
    pred_len = len(pred_code)

    # ---- 字符级比对（标签几位比几位）----
    # 对于标签中的每个位置：
    # - 如果预测该位置有字符且不是 ? → 比较
    # - 如果预测该位置是 ? 或缺失 → 计为错误
    for pos in range(gt_len):
        gt_char = gt_chars[pos]

        # 只在 0-3 位范围内统计位置准确率
        if pos < 4:
            pos_total[pos] += 1

        if pos < pred_len and pred_code[pos] != '?':
            pred_char = pred_code[pos]
            if gt_char == pred_char:
                total_correct_chars += 1
                if pos < 4:
                    pos_correct[pos] += 1
            else:
                # 替换错误
                if pos < 4:
                    confusion.append((pos, gt_char, pred_char))
                else:
                    confusion.append((-1, gt_char, pred_char))
        else:
            # 预测缺失该位置 → 删除错误
            if pos < 4:
                confusion.append((pos, gt_char, '(缺失)'))
            else:
                confusion.append((-1, gt_char, '(缺失)'))

    # ---- 编辑距离（基于完整字符串）----
    ed = levenshtein(pred_code, gt_code)
    total_edit_distance += ed

    # ---- 4 位完全匹配率 ----
    if gt_len == 4 and pred_len == 4:
        if gt_code == pred_code:
            exact_match_4 += 1

    # ---- 记录错误样本 ----
    if pred_code != gt_code:
        errors.append({
            'filename': img_name,
            'gt': gt_code,
            'pred': pred_code if pred_code else '(空)',
            'reason': f'GT{gt_len}位/Pred{pred_len}位, ED={ed}'
        })

# ========== 6. 输出报告 ==========
print("\n" + "=" * 60)
print("准确率评估报告（标准 OCR 评估口径）")
print("=" * 60)
print(f"\n总评估样本数: {total_samples}")
print(f"标签总字符数: {total_gt_chars}")

print("\n--- 核心指标 ---")
car = total_correct_chars / total_gt_chars * 100 if total_gt_chars > 0 else 0
cer = (total_edit_distance / total_gt_chars) * 100 if total_gt_chars > 0 else 0
print(f"CAR (字符准确率): {total_correct_chars}/{total_gt_chars} = {car:.2f}%")
print(f"CER (字符错误率): {total_edit_distance}/{total_gt_chars} = {cer:.2f}%")

print("\n各位置准确率（仅统计 0-3 位）:")
pos_names = ["首位", "第2位", "第3位", "第4位"]
for i in range(4):
    if pos_total[i] > 0:
        acc = pos_correct[i] / pos_total[i] * 100
        print(f"  {pos_names[i]}: {pos_correct[i]}/{pos_total[i]} = {acc:.2f}%")

print(f"\n4 位完全匹配率（仅 {images_with_4_gt} 张 4 位标注图）: "
      f"{exact_match_4}/{images_with_4_gt} = "
      f"{(exact_match_4 / images_with_4_gt * 100):.2f}%" if images_with_4_gt > 0 else "N/A")

print("\n--- 混淆详情 (位置, 真实->识别, 次数) ---")
if confusion:
    cc = defaultdict(int)
    for pos, gt, pred in confusion:
        cc[(pos, gt, pred)] += 1
    # 按出现次数降序，最多显示 30 条
    for (pos, gt, pred), cnt in sorted(cc.items(), key=lambda x: -x[1])[:30]:
        if pos == -1:
            pname = "未知位置"
        else:
            pname = pos_names[pos]
        print(f"  {pname}: {gt} -> {pred}  (出现 {cnt} 次)")
else:
    print("  ✅ 无错误！")

print(f"\n错误样本数: {len(errors)}")
if errors:
    print("前 20 个样本:")
    for e in errors[:20]:
        print(f"  {e['filename']}: GT={e['gt']} Pred={e['pred']} ({e['reason']})")

    errors_df = pd.DataFrame(errors)
    errors_df.to_csv(BASE_DIR / "evaluation_errors.csv", index=False, encoding="utf-8-sig")
    print(f"\n错误样本已保存: {BASE_DIR / 'evaluation_errors.csv'}")

# ========== 7. 额外：每个字符的准确率 ==========
print("\n" + "-" * 60)
print("按真实字符统计准确率:")
char_stats = defaultdict(lambda: [0, 0])
for txt_path in txt_files:
    gt_chars = parse_label_variable_len(txt_path)
    if not gt_chars:
        continue
    img_name = txt_path.stem + ".jpg"
    pred_code = results_dict.get(img_name, "") or ""

    for pos, gt_c in enumerate(gt_chars):
        # 预测对应位置
        if pos < len(pred_code):
            pred_c = pred_code[pos]
            char_stats[gt_c][1] += 1
            if pred_c != '?' and gt_c == pred_c:
                char_stats[gt_c][0] += 1
        else:
            char_stats[gt_c][1] += 1

print("\n字符 | 正确数/总数 | 准确率")
print("-" * 40)
for char in sorted(char_stats.keys()):
    correct, total = char_stats[char]
    if total > 0:
        acc = correct / total * 100
        bar = "█" * int(acc / 5)
        print(f"  {char:>3}: {correct:>3}/{total:<3} = {acc:5.1f}%  {bar}")