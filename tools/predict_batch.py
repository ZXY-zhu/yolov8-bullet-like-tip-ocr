"""
tools/predict_batch.py
批量端到端识别：遍历图片文件夹 → YOLO 检测 → 预处理 → OCR → 按位置纠错 → 导出 CSV

★ 编号规则（固化）：
  - 4 个框：首位大写字母 + 后三位数字(0-9)
  - 少于 4 个框：因为首位字母在最底部，高曝光时最先消失，
    所以剩余可见字符全部按数字处理
  - 0 个框：空样本（器件旋转/未拍到刻印），合法结果

★ 性能：实时打印速度和预估剩余时间，方便评估 1813 张总耗时
"""
# ★ 必须在 import paddleocr 之前设置，关闭 oneDNN（规避 PIR 崩溃）
import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import time
from paddleocr import PaddleOCR
from ultralytics import YOLO

# ========== 路径配置 ==========
BASE_DIR = Path(r"C:\Users\BeLig\PycharmProjects\yolov8-bullet-like-tip-ocr")
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

IMAGE_DIR = Path(r"C:\Users\BeLig\Desktop\dataset\images")
OUTPUT_CSV = BASE_DIR / "recognition_results.csv"

YOLO_WEIGHTS = str(BASE_DIR / "runs" / "detect" / "bullet_tip_v1" / "weights" / "best.pt")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ========== 加载模型（只加载一次）==========
print("加载 YOLO...")
model = YOLO(YOLO_WEIGHTS)

print("加载 OCR...")
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
    enable_mkldnn=False,
)
print("就绪。\n")

# ========== 字符纠错映射 ==========
# 数字位的形近映射（所有数字位共用）
DIGIT_MAP = {
    'O': '0', 'o': '0', 'D': '0', 'Q': '0',
    'I': '1', 'l': '1', '|': '1',
    'S': '5', 's': '5',
    'B': '8',
    'Z': '2',
    'A': '4',
    'G': '6', 'T': '7', 'E': '3', 'P': '9',
}

# 字母位的形近映射（仅 4 框时的首位使用）
LETTER_MAP = {
    '0': 'D', 'O': 'D', 'Q': 'D',
    '1': 'I', 'l': 'I', '|': 'I',
    '5': 'S',
    '8': 'B',
    '2': 'Z',
    '4': 'A',
    '6': 'G', '7': 'T', '3': 'E', '9': 'P',
}

# 合法字符集
VALID_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def process_single_image(img_path: Path, save_debug=False):
    """
    处理单张图片，返回变长编号字符串（0~4位）。

    规则：
    - 0 个框 → ("", "EMPTY")
    - 4 个框 → 首位字母 + 后三位数字
    - 1~3 个框 → 全部按数字处理（因为首位字母最先因曝光消失）
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None, "图片读取失败"

    results = model(img, conf=0.25, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # 0 框 = 空样本
    if len(boxes) == 0:
        return "", "EMPTY"

    # 从下到上排序（y1 大的在下方，首位字母在最下面）
    boxes = sorted(boxes, key=lambda b: b[1], reverse=True)
    boxes = boxes[:4]  # 最多 4 个框

    num_boxes = len(boxes)
    # ★ 关键：只有 4 框时，第 0 位才是字母；否则所有位都按数字处理
    has_letter = (num_boxes == 4)

    recognized = []
    raw_texts = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        h_img, w_img = img.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            recognized.append("?")
            raw_texts.append("")
            continue

        # ---- 预处理 ----
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) > 127:
            gray = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.sum(binary == 255) > np.sum(binary == 0):
            binary = cv2.bitwise_not(binary)
        kernel = np.ones((3, 3), np.uint8)
        morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        char_upright = cv2.rotate(morphed, cv2.ROTATE_90_CLOCKWISE)
        if np.sum(char_upright == 255) > np.sum(char_upright == 0):
            char_upright = cv2.bitwise_not(char_upright)

        # ---- OCR ----
        char_3ch = cv2.cvtColor(char_upright, cv2.COLOR_GRAY2BGR)
        result = ocr.predict(char_3ch)

        char_text = "?"
        raw_text = ""
        try:
            for res in result:
                # 兼容两种返回结构
                if isinstance(res, dict):
                    texts = res.get("rec_texts", [])
                    scores = res.get("rec_scores", [])
                else:
                    # 结果对象
                    texts = getattr(res, "rec_texts", [])
                    scores = getattr(res, "rec_scores", [])

                if texts:
                    raw_text = texts[0].strip()
                    # 提取第一个合法字符
                    for c in raw_text:
                        if c in VALID_CHARS:
                            char_text = c
                            break
                    break
        except Exception as e:
            raw_text = f"OCR异常: {e}"

        # ==================== 按位置纠错 ====================
        if has_letter and i == 0:
            # ★ 4 框时的首位：大写字母
            if char_text.isalpha():
                char_text = char_text.upper()
            elif char_text in LETTER_MAP:
                char_text = LETTER_MAP[char_text]
            # 否则保留原始结果（可能是 ? 或其他）
        else:
            # ★ 其余所有情况：数字
            if char_text.isdigit():
                pass  # 已经是数字
            elif char_text in DIGIT_MAP:
                char_text = DIGIT_MAP[char_text]
            # 否则保留原始结果

        recognized.append(char_text)
        raw_texts.append(raw_text)

        if save_debug:
            prefix = f"{img_path.stem}_char_{i}"
            cv2.imwrite(str(ASSETS_DIR / f"{prefix}_upright.png"), char_upright)

    code = "".join(recognized)
    return code, "|".join(raw_texts)


# ========== 批量处理 ==========
print(f"扫描图片目录: {IMAGE_DIR}")
image_files = [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
image_files.sort()
total = len(image_files)
print(f"找到 {total} 张图片\n")

results = []
success_count = 0
fail_count = 0
box_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

# ★ 测速
start_time = time.time()
last_print_time = start_time

for idx, img_path in enumerate(image_files, 1):
    try:
        code, detail = process_single_image(img_path, save_debug=False)

        if code is None:
            fail_count += 1
            print(f"[{idx}/{total}] {img_path.name}: ✗ {detail}")
            results.append({
                "filename": img_path.name, "code": "",
                "status": "ERROR", "detail": detail, "num_boxes": 0,
            })
        elif detail == "EMPTY":
            success_count += 1
            box_dist[0] += 1
            results.append({
                "filename": img_path.name, "code": "",
                "status": "EMPTY", "detail": "0个框", "num_boxes": 0,
            })
        else:
            success_count += 1
            num_boxes = len(code)
            box_dist[num_boxes] += 1
            has_unknown = "?" in code
            status = "OK" if (num_boxes == 4 and not has_unknown) else "PARTIAL"
            results.append({
                "filename": img_path.name, "code": code,
                "status": status, "detail": detail, "num_boxes": num_boxes,
            })

            # ★ 每张都打印（带 flush 确保立即输出到终端）
            if idx % 10 == 0 or idx == total:  # 每10张打印一次，兼顾性能和可见性
                now = time.time()
                elapsed = now - start_time
                avg_per_img = elapsed / idx
                remaining = avg_per_img * (total - idx)

                if detail != "EMPTY" and code is not None:
                    brief = f"'{code}'"
                elif detail == "EMPTY":
                    brief = "(空)"
                else:
                    brief = "✗"

                print(f"[{idx}/{total}] {img_path.name}: {brief}  "
                      f"速度: {avg_per_img:.2f}秒/张 | "
                      f"已用: {elapsed / 60:.1f}分钟 | "
                      f"预计剩余: {remaining / 60:.1f}分钟", flush=True)

    except Exception as e:
        fail_count += 1
        print(f"[{idx}/{total}] {img_path.name}: ✗ 异常 {e}")
        results.append({
            "filename": img_path.name, "code": "",
            "status": "ERROR", "detail": str(e), "num_boxes": 0,
        })

# ========== 导出 CSV ==========
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# ========== 汇总 ==========
total_elapsed = time.time() - start_time
print("\n" + "=" * 60)
print(f"批量识别完成！总耗时: {total_elapsed/60:.1f}分钟")
print(f"总图片数: {total}")
print(f"成功识别: {success_count}")
print(f"失败/异常: {fail_count}")

ok_count = sum(1 for r in results if r['status'] == 'OK')
partial_count = sum(1 for r in results if r['status'] == 'PARTIAL')
empty_count = sum(1 for r in results if r['status'] == 'EMPTY')
error_count = sum(1 for r in results if r['status'] == 'ERROR')

print(f"\n完整识别（4位无?）: {ok_count}")
print(f"部分识别（1~3位或有?）: {partial_count}")
print(f"空样本（0框/未拍到刻印）: {empty_count}")
print(f"异常失败: {error_count}")

print(f"\n识别框数量分布:")
for k in range(5):
    if box_dist[k] > 0:
        if k == 4:
            desc = " (4位: 首位字母+三位数字)"
        elif k == 0:
            desc = " (空样本)"
        else:
            desc = f" ({k}位: 全数字)"
        print(f"  {k} 个框: {box_dist[k]} 张{desc}")

print(f"\n结果已保存: {OUTPUT_CSV}")
print("=" * 60)

errors = [r for r in results if r['status'] == 'ERROR']
if errors:
    print(f"\n异常图片（共 {len(errors)} 张）:")
    for r in errors[:20]:
        print(f"  - {r['filename']}: {r['detail']}")