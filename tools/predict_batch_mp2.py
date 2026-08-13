"""
predict_batch_mp2.py
2 进程并行批量识别（适用于 16GB 内存机器）

★ 关键设计：
  - 主进程不加载 PaddleOCR，避免 Windows spawn 冲突
  - 每个子进程通过 initializer 独立加载 YOLO + PaddleOCR
  - 结果统一收集后写 CSV，避免并发写文件冲突
"""
import os

# ★ 必须在 import paddleocr 之前设置
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import time
from multiprocessing import Pool
from paddleocr import PaddleOCR
from ultralytics import YOLO

# ========== 路径配置 ==========
BASE_DIR = Path(r"C:\Users\BeLig\PycharmProjects\yolov8-bullet-like-tip-ocr")
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

IMAGE_DIR = Path(r"C:\Users\BeLig\Desktop\dataset\images")
OUTPUT_CSV = BASE_DIR / "recognition_results_mp2.csv"  # 用新文件名，不与单进程结果冲突

YOLO_WEIGHTS = str(BASE_DIR / "runs" / "detect" / "bullet_tip_v1" / "weights" / "best.pt")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
NUM_WORKERS = 2  # ★ 2 进程，16GB 内存安全值

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

# 全局变量，由 initializer 在每个子进程里赋值
model = None
ocr = None


def init_worker():
    """每个子进程启动时独立加载模型（避免 Windows spawn 冲突）"""
    global model, ocr
    print(f"[Worker {os.getpid()}] 加载 YOLO...", flush=True)
    model = YOLO(YOLO_WEIGHTS)
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
    """子进程处理函数：处理单张图片并返回结果元组"""
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

        # 从下到上排序
        boxes = sorted(boxes, key=lambda b: b[1], reverse=True)
        boxes = boxes[:4]

        num_boxes = len(boxes)
        has_letter = (num_boxes == 4)  # 只有4框时首位才是字母

        recognized = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            h_img, w_img = img.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)

            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                recognized.append("?")
                continue

            # 预处理
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
                # 首位字母
                if char_text.isalpha():
                    char_text = char_text.upper()
                elif char_text in LETTER_MAP:
                    char_text = LETTER_MAP[char_text]
            else:
                # 数字位（4框的后三位，或1~3框的所有位）
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
    print(f"扫描图片目录: {IMAGE_DIR}")
    image_files = [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    image_files.sort()
    total = len(image_files)
    print(f"找到 {total} 张图片，使用 {NUM_WORKERS} 进程并行处理\n")

    # 转换为字符串路径列表传给子进程
    img_path_strs = [str(p) for p in image_files]

    start_time = time.time()

    # ★ 创建进程池，每个子进程独立加载模型
    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        print("进程池已启动，开始并行处理...\n", flush=True)

        # 使用 imap_unordered 可以边处理边获取结果
        results = []
        completed = 0

        for result in pool.imap_unordered(process_single_image, img_path_strs, chunksize=10):
            results.append(result)
            completed += 1

            # 每 50 张打印一次进度
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

    # ========== 汇总结果 ==========
    total_elapsed = time.time() - start_time

    # 按文件名排序，保证输出顺序与输入一致
    results.sort(key=lambda x: x[0])

    # 转换为 DataFrame 格式
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

    # ========== 打印汇总 ==========
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
            if k == 4:
                desc = " (4位: 首位字母+三位数字)"
            elif k == 0:
                desc = " (空样本)"
            else:
                desc = f" ({k}位: 全数字)"
            print(f"  {k} 个框: {box_dist[k]} 张{desc}")

    print(f"\n结果已保存: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    # Windows 下必须使用 spawn 启动方式，避免 PaddleOCR 多进程冲突
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    main()