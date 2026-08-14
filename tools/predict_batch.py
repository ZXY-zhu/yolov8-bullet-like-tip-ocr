"""
tools/predict_batch.py
2 进程并行批量识别（适用于 16GB RAM 机器）
★ v2 预处理管道（目前最高）
★ 输出：tools/outputs/recognition_results_v2_full.csv
"""
import os
import cv2
import numpy as np
import pandas as pd
import time
from pathlib import Path
from multiprocessing import Pool
from paddleocr import PaddleOCR
from ultralytics import YOLO

# ★ 必须在 import paddleocr 之前设置，关闭 oneDNN（规避 PIR 崩溃）
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

# ========== 路径配置（修改此处） ==========
# 脚本所在目录设为 BASE_DIR (即 tools/)
BASE_DIR = Path(__file__).parent.resolve()

# 1. 图片输入目录：改成你的数据集图片路径
IMAGE_DIR = Path(r"C:\Users\BeLig\Desktop\dataset\images")  # ← 修改此处

# 2. 模型权重路径：改成你的 best.pt 路径
YOLO_WEIGHTS = BASE_DIR.parent / "runs" / "detect" / "bullet_tip_v1" / "weights" / "best.pt"  # ← 修改此处（如果 best.pt 不在项目根目录 runs 下，请改绝对路径）

# 3. 输出 CSV 路径：默认放在 tools/outputs/ 下
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)  # 自动创建 outputs 文件夹（如果 .gitignore 允许）
OUTPUT_CSV = OUTPUT_DIR / "recognition_results_v2_full.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
NUM_WORKERS = 2  # 根据 CPU 核心数调整

# ========== 字符纠错映射 ==========
DIGIT_MAP = {
    'O': '0', 'o': '0', 'D': '0', 'Q': '0',
    'I': '1', 'l': '1', '|': '1',
    'S': '5', 's': '5',
    'B': '8', 'Z': '2', 'A': '4',
    'G': '6', 'T': '7', 'E': '3', 'P': '9',
}
LETTER_MAP = {
    '0': 'D', 'O': 'D', 'Q': 'D',
    '1': 'I', 'l': 'I', '|': 'I',
    '5': 'S', '8': 'B', '2': 'Z', '4': 'A',
    '6': 'G', '7': 'T', '3': 'E', '9': 'P',
}
VALID_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

# 全局变量
model = None
ocr = None

def resize_and_pad(image, target_height=48, target_width=320):
    h, w = image.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
    pad_w = target_width - new_w
    if pad_w > 0:
        padded = cv2.copyMakeBorder(resized, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    else:
        padded = cv2.resize(resized, (target_width, target_height))
    return padded

def init_worker():
    global model, ocr
    print(f"[Worker {os.getpid()}] 加载 YOLO...", flush=True)
    # 注意：YOLO 接收字符串路径
    model = YOLO(str(YOLO_WEIGHTS))

    print(f"[Worker {os.getpid()}] 加载 OCR...", flush=True)
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
        enable_mkldnn=False,
    )
    print(f"[Worker {os.getpid()}] 模型加载完成 ✅", flush=True)

def process_single_image(img_path_str: str):
    global model, ocr
    img_path = Path(img_path_str)

    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return (img_path.name, "", "ERROR", "图片读取失败", 0)

        results = model(img, conf=0.25, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            return (img_path.name, "", "EMPTY", "0个框", 0)

        boxes = sorted(boxes, key=lambda b: b[1], reverse=True)
        boxes = boxes[:4]  # 只取前 4 个

        num_boxes = len(boxes)
        has_letter = (num_boxes == 4)

        recognized = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            h_img, w_img = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                recognized.append("?")
                continue

            # ========== 预处理管道 v2（目前最高）==========
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) > 127:
                gray = cv2.bitwise_not(gray)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            blurred = cv2.GaussianBlur(enhanced, (9, 9), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if np.sum(binary == 255) > np.sum(binary == 0):
                binary = cv2.bitwise_not(binary)

            solid_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            solid = cv2.dilate(binary, solid_kernel, iterations=1)

            denoise_kernel = np.ones((3, 3), np.uint8)
            morphed = cv2.morphologyEx(solid, cv2.MORPH_OPEN, denoise_kernel, iterations=1)

            char_upright = cv2.rotate(morphed, cv2.ROTATE_90_CLOCKWISE)
            if np.sum(char_upright == 255) > np.sum(char_upright == 0):
                char_upright = cv2.bitwise_not(char_upright)

            char_upright = resize_and_pad(char_upright, target_height=48, target_width=320)

            # OCR
            char_3ch = cv2.cvtColor(char_upright, cv2.COLOR_GRAY2BGR)
            result = ocr.predict(char_3ch)

            char_text = "?"
            for res in result:
                texts = res.get("rec_texts", [])
                if texts:
                    raw_text = texts[0].strip()
                    for c in raw_text:
                        if c in VALID_CHARS:
                            char_text = c
                            break
                    break

            # 按位置纠错
            if has_letter and i == 0:
                # 首位：必须是字母
                if char_text.isalpha():
                    char_text = char_text.upper()
                elif char_text in LETTER_MAP:
                    char_text = LETTER_MAP[char_text]
            else:
                # 后三位：必须是数字
                if char_text.isdigit():
                    pass
                elif char_text in DIGIT_MAP:
                    char_text = DIGIT_MAP[char_text]

            recognized.append(char_text)

        code = "".join(recognized)
        has_unknown = "?" in code
        status = "OK" if (num_boxes == 4 and not has_unknown) else "PARTIAL"

        return (img_path.name, code, status, "", num_boxes)

    except Exception as e:
        return (img_path.name, "", "ERROR", str(e), 0)

def main():
    # 检查路径是否存在
    if not IMAGE_DIR.exists():
        print(f"❌ 错误：图片目录不存在 -> {IMAGE_DIR}")
        print("   请修改脚本顶部的 IMAGE_DIR 变量为你的图片路径")
        return
    if not YOLO_WEIGHTS.exists():
        print(f"❌ 错误：模型权重不存在 -> {YOLO_WEIGHTS}")
        print("   请修改脚本顶部的 YOLO_WEIGHTS 变量为 best.pt 的路径")
        return

    image_files = [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    image_files.sort()
    total = len(image_files)

    if total == 0:
        print(f"⚠️ 警告：在 {IMAGE_DIR} 中未找到图片")
        return

    print(f"扫描图片目录: {IMAGE_DIR}")
    print(f"找到 {total} 张图片，使用 {NUM_WORKERS} 进程并行处理\n")
    print(f"模型路径: {YOLO_WEIGHTS}")
    print(f"结果将保存至: {OUTPUT_CSV}\n")

    img_path_strs = [str(p) for p in image_files]

    start_time = time.time()

    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        print("进程池已启动，开始并行处理...\n", flush=True)

        results = []
        completed = 0

        for result in pool.imap_unordered(process_single_image, img_path_strs, chunksize=10):
            results.append(result)
            completed += 1

            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - start_time
                avg_per_img = elapsed / completed
                remaining = avg_per_img * (total - completed)
                fname, code, status, detail, nbox = result
                brief = f"'{code}'" if code else f"({status})"
                print(f"[{completed}/{total}] {fname}: {brief}  "
                      f"速度: {avg_per_img:.2f}秒/张 | "
                      f"已用: {elapsed / 60:.1f}分钟 | "
                      f"预计剩余: {remaining / 60:.1f}分钟", flush=True)

    total_elapsed = time.time() - start_time
    results.sort(key=lambda x: x[0])

    df_data = []
    box_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for fname, code, status, detail, nbox in results:
        df_data.append({
            "filename": fname,
            "code": code,
            "status": status,
            "detail": detail,
            "num_boxes": nbox,
        })
        box_dist[nbox] = box_dist.get(nbox, 0) + 1

    df = pd.DataFrame(df_data)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print(f"批量识别完成！总耗时: {total_elapsed / 60:.1f}分钟（{NUM_WORKERS}进程并行）")
    print(f"总图片数: {total}")

    ok_count = sum(1 for r in df_data if r['status'] == 'OK')
    partial_count = sum(1 for r in df_data if r['status'] == 'PARTIAL')
    empty_count = sum(1 for r in df_data if r['status'] == 'EMPTY')
    error_count = sum(1 for r in df_data if r['status'] == 'ERROR')

    print(f"\n完整识别（4位无?）: {ok_count}")
    print(f"部分识别（1~3位或有?）: {partial_count}")
    print(f"空样本（0框/未拍到刻印）: {empty_count}")
    print(f"异常失败: {error_count}")

    print(f"\n识别框数量分布:")
    for k in range(5):
        if box_dist[k] > 0:
            desc = " (空样本)" if k == 0 else f" ({k}位: {'首位字母+后三位数字' if k == 4 else '全数字'})"
            print(f"  {k} 个框: {box_dist[k]} 张{desc}")

    print(f"\n结果已保存: {OUTPUT_CSV}")
    print("=" * 60)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()