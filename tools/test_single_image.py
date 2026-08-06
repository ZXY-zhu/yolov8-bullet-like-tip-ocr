from paddleocr import PaddleOCR
import cv2

img_path = "C:/Users/BeLig/....jpg"  # 换成你的一张真实图片

# PaddleOCR 初始化
ocr = PaddleOCR(
    use_textline_orientation=True,
    lang="en",
    enable_mkldnn=False   # 禁用 oneDNN/PIR 崩溃路径
)

# 读图
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"图片读取失败: {img_path}")

# 推理
result = ocr.predict(img)

# 解析结果
for res in result:
    texts = res.get("rec_texts", [])
    boxes = res.get("rec_boxes", [])
    print(f"检测到 {len(texts)} 个文本区域")

    for text, score in zip(res.get("rec_texts", []), res.get("rec_scores", [])):
        print(f"识别结果: {text}, 置信度: {score:.2f}")