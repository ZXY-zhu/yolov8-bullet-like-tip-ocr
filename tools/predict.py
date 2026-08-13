"""
tools/predict.py
端到端：YOLO 检测 → 裁剪 → 预处理 → 旋转 → 实心化 → OCR → 按位置纠错 → 输出编号
编号规则：首位大写字母 + 后三位数字(0-9)
"""

# ★ 必须在 import paddleocr 之前设置，关闭 oneDNN（规避 PIR 崩溃）
import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

import cv2
import numpy as np
from pathlib import Path
from paddleocr import PaddleOCR
from ultralytics import YOLO

# ========== 路径 ==========
BASE_DIR = Path(r"C:\Users\BeLig\PycharmProjects\yolov8-bullet-like-tip-ocr")
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

IMG_PATH = r"C:\Users\BeLig\Desktop\dataset\images\20260716_201144_799_4.jpg"
YOLO_WEIGHTS = str(BASE_DIR / "runs" / "detect" / "bullet_tip_v1" / "weights" / "best.pt")

# ========== 加载模型 ==========
print("加载 YOLO...")
model = YOLO(YOLO_WEIGHTS)

print("加载 OCR...")
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
    enable_mkldnn=False,    # 双保险之一：绕过 oneDNN/PIR 崩溃
)
print("就绪。\n")

# ========== 读图 + 检测 ==========
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"图片读取失败: {IMG_PATH}")

print(f"图片: {IMG_PATH}  ({img.shape[1]}x{img.shape[0]})")

results = model(img, conf=0.25, verbose=False)
boxes = results[0].boxes.xyxy.cpu().numpy()
boxes = sorted(boxes, key=lambda b: b[1], reverse=True)  # 从下到上

print(f"检测到 {len(boxes)} 个框\n")
print("=" * 60)

recognized = []

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
    h_img, w_img = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        print(f"[{i}] ROI 为空")
        recognized.append("?")
        continue

    # ---- 预处理（沿用你能识别 4 的版本）----
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 自动明暗反转：确保背景暗、字符亮
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # OTSU 二值化
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 确保字符是白色
    if np.sum(binary == 255) > np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    # 开运算去噪
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 旋转转正
    char_upright = cv2.rotate(morphed, cv2.ROTATE_90_CLOCKWISE)

    # 统一为黑底白字
    if np.sum(char_upright == 255) > np.sum(char_upright == 0):
        char_upright = cv2.bitwise_not(char_upright)

    # 轮廓实心化：把空心 D/0/4 填成实心
    contours, _ = cv2.findContours(char_upright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_u, w_u = char_upright.shape
    min_area = (h_u * w_u) * 0.015
    valid = [c for c in contours if cv2.contourArea(c) > min_area]
    solid = np.zeros_like(char_upright)
    if valid:
        cv2.drawContours(solid, valid, -1, 255, thickness=cv2.FILLED)
    else:
        solid = char_upright

    # 转成白底黑字送 OCR（PaddleOCR 训练数据主要分布）
    ocr_input_gray = cv2.bitwise_not(solid)

    # ---- 保存调试 ----
    cv2.imwrite(str(ASSETS_DIR / f"char_{i}_raw.png"), roi)
    cv2.imwrite(str(ASSETS_DIR / f"char_{i}_binary.png"), binary)
    cv2.imwrite(str(ASSETS_DIR / f"char_{i}_morphed.png"), morphed)
    cv2.imwrite(str(ASSETS_DIR / f"char_{i}_solid.png"), solid)
    cv2.imwrite(str(ASSETS_DIR / f"char_{i}_ocr_input.png"), ocr_input_gray)

    # ---- OCR ----
    char_3ch = cv2.cvtColor(ocr_input_gray, cv2.COLOR_GRAY2BGR)
    result = ocr.predict(char_3ch)

    char_text = "?"
    char_score = 0.0
    raw_text = ""
    for res in result:
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        if texts:
            raw_text = texts[0].strip()
            # 提取第一个合法字符（字母或数字）
            for c in raw_text:
                if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
                    char_text = c
                    char_score = scores[0]
                    break
            break

    # ==================== 按位置纠错 ====================
    # 首位：大写字母；后三位：数字 0-9
    # 字母位的形近映射（首位专用）
    LETTER_MAP = {
        '0': 'D', 'O': 'D', 'Q': 'D',         # 最可能是 D
        '1': 'I', 'l': 'I', '|': 'I',          # → I
        '5': 'S', 'S': 'S',
        '8': 'B', 'B': 'B',
        '2': 'Z', 'Z': 'Z',
        '4': 'A', 'A': 'A',
        '6': 'G', 'G': 'G',
        '7': 'T', 'T': 'T',
        '3': 'E', 'E': 'E',
        '9': 'P', 'P': 'P',
    }

    # 数字位的形近映射（后三位专用）
    DIGIT_MAP = {
        'O': '0', 'o': '0', 'D': '0', 'Q': '0',
        'I': '1', 'l': '1', '|': '1',
        'S': '5', 's': '5',
        'B': '8',
        'Z': '2',
        'A': '4',
        'G': '6',
        'T': '7',
        'E': '3',
        'P': '9',
    }

    if i == 0:
        # ★ 首位：必须是大写字母
        if char_text.isalpha():
            char_text = char_text.upper()
        elif char_text in LETTER_MAP:
            char_text = LETTER_MAP[char_text]
            print(f"  [{i}] 首位 '{raw_text}' -> 修正为字母 '{char_text}'")
        else:
            # 不在映射表中，保留原始结果
            print(f"  [{i}] 首位 '{char_text}' 不在字母白名单，保留原始识别")
    else:
        # ★ 后三位：必须是数字
        if char_text.isdigit():
            pass  # 已经是数字，不动
        elif char_text in DIGIT_MAP:
            char_text = DIGIT_MAP[char_text]
            print(f"  [{i}] 第{i+1}位 '{raw_text}' -> 修正为数字 '{char_text}'")
        else:
            print(f"  [{i}] 第{i+1}位 '{char_text}' 不在数字白名单，保留原始识别")

    # ====================================================

    recognized.append(char_text)
    print(f"[{i}] {char_text}  ({char_score:.4f})  raw='{raw_text}'")

# ========== 输出 ==========
print("\n" + "=" * 60)
print(f"识别结果: {''.join(recognized)}")
print(f"调试图片: {ASSETS_DIR}")