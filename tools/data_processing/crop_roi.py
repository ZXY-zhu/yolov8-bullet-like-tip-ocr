"""
tools/data_processing/crop_roi.py
用途：用 YOLOv8 检测刻印区域，裁剪 ROI，送 PaddleOCR 识别
说明：
- 本脚本为端到端验证用，不负责训练
- 依赖：ultralytics, paddlepaddle, paddleocr, opencv-python
- 环境：CPU / Python 3.10
"""

from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2

# ========== 1. 配置区 ==========
img_path = "test.jpg"   # 改为实际测试图片路径
model_path = "weights/best.pt"   # 改为实际模型权重路径

conf_thres = 0.25                               # YOLO 置信度阈值
ocr_lang = "en"

# ========== 2. 初始化模型 ==========
# YOLO：加载训练好的刻印检测模型
model = YOLO(model_path)

# PaddleOCR：初始化识别引擎
ocr = PaddleOCR(
    use_textline_orientation=True,
    lang=ocr_lang,
    enable_mkldnn=False
)

# ========== 3. 读图 ==========
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"图片读取失败: {img_path}")

# ========== 4. YOLO 检测 ==========
# verbose=False：不打印训练日志
results = model(img, conf=conf_thres, verbose=False)

# results[0].boxes.xyxy: 检测框坐标 (x1, y1, x2, y2)
boxes = results[0].boxes.xyxy.cpu().numpy()

print(f"YOLO 检测到 {len(boxes)} 个候选区域")

# ========== 5. 裁剪 ROI 并 OCR ==========
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)

    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        continue

    ocr_result = ocr.predict(roi)

    for res in ocr_result:
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        for text, score in zip(texts, scores):
            print(f"ROI[{i}] 识别结果: {text}, 置信度: {score:.2f}")