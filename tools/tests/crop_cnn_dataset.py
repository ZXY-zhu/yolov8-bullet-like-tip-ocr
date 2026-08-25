"""
tools/tests/crop_cnn_dataset.py
用途：从 YOLO 格式标注中按检测框裁剪 ROI，转灰度并 padding 到正方形后 resize 为 48×48，按类别保存
说明：
- 输入：YOLO txt 标注 + 原始图片（.jpg）
- 处理：裁剪 → 灰度 → padding 到正方形 → resize 48×48
- 输出：按类别名命名的文件夹，每个文件夹内存放对应字符的 48×48 灰度 PNG
- 类别映射：从 classes.txt 读取
- 运行方式：直接改下面 4 个路径，然后 python tools/tests/crop_cnn_dataset.py
"""

import cv2
import numpy as np
import os

# ========== 改这里 ==========
classes_path = "labels_raw/classes.txt"               # 改为实际 classes.txt 路径
labels_dir = "labels_raw"                            # 改为实际标注目录路径
images_dir = "dataset/images"                        # 改为实际图片目录路径
save_root = "dataset_cnn"                            # 改为实际输出目录路径
# ============================

# ========== 1. 读 classes.txt ==========
with open(classes_path, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]
print(f"类别数量: {len(class_names)}")

# ========== 2. 统计计数器 ==========
counts = {}

# ========== 3. 遍历所有标签文件 ==========
for txt_name in os.listdir(labels_dir):

    if not txt_name.endswith(".txt") or txt_name == "classes.txt":
        continue

    txt_path = os.path.join(labels_dir, txt_name)
    img_name = txt_name.replace(".txt", ".jpg")
    img_path = os.path.join(images_dir, img_name)

    print(f"处理中: {img_name}", end="\r")

    if not os.path.exists(img_path):
        print(f"⚠️ 跳过，找不到图片: {img_name}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 跳过，读不了图片: {img_name}")
        continue

    img_h, img_w = img.shape[:2]

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # 坐标还原
        x_center_px = x_center * img_w
        y_center_px = y_center * img_h
        w_px = width * img_w
        h_px = height * img_h

        x1 = int(x_center_px - w_px / 2)
        y1 = int(y_center_px - h_px / 2)
        x2 = int(x_center_px + w_px / 2)
        y2 = int(y_center_px + h_px / 2)

        # 边界保护
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # 转灰度
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Padding 到正方形
        side = max(gray.shape[0], gray.shape[1])
        square = np.zeros((side, side), dtype=np.uint8)
        y_offset = (side - gray.shape[0]) // 2
        x_offset = (side - gray.shape[1]) // 2
        square[y_offset:y_offset + gray.shape[0], x_offset:x_offset + gray.shape[1]] = gray

        # Resize 48×48
        resized = cv2.resize(square, (48, 48))

        # 字符名
        char_name = class_names[class_id]

        # 创建字符文件夹
        char_dir = os.path.join(save_root, char_name)
        os.makedirs(char_dir, exist_ok=True)

        # 保存
        save_name = f"{img_name.replace('.jpg', '')}_{class_id}.png"
        cv2.imwrite(os.path.join(char_dir, save_name), resized)

        # 计数
        counts[char_name] = counts.get(char_name, 0) + 1

# ========== 4. 打印统计 ==========
print("\n🎉 裁剪完成！每个字符的数量：")
for char, count in sorted(counts.items()):
    print(f"  {char}: {count} 张")
print(f"\n总计: {sum(counts.values())} 张")