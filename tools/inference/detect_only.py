"""
tools/inference/detect_only.py
用途：只跑 YOLO 检测，输出框（类别全 0）
说明：
- 输入：待检测图片目录
- 输出：YOLO 格式 txt 标注（所有类别 ID 强制写为 0）+ classes.txt
- 运行方式：直接改下面 3 个路径，然后 python tools/detect_only.py
"""

## 【数据流】 导入依赖。Path 处理路径，cv2 读图，YOLO 是检测模型。
from pathlib import Path
import cv2
from ultralytics import YOLO

## 【为什么】 这 3 个路径 + 2 个超参是全局配置，改一处就能换数据集/换权重/调阈值，不用动下面逻辑。
# ========== 改这里 ==========
IMAGES_DIR = Path("images")                                    # 改为实际待检测图片目录路径
WEIGHTS = "weights/best.pt"                                    # 改为实际模型权重路径
OUT_DIR = Path("labels_pred")                                  # 改为实际输出目录路径
CONF = 0.25
IMGSZ = 640
# ============================

## 【数据流】 创建输出目录。parents=True 递归创建父目录（缺哪层建哪层），exist_ok=True 已存在不报错。
OUT_DIR.mkdir(parents=True, exist_ok=True)

## 【数据流】 写 classes.txt。只写一行 "tip_number_region"，因为只检测一类（刻印区域）。
##          LabelImg 打开标注时需要这个文件知道类别名。
with open(OUT_DIR / "classes.txt", "w") as f:
    f.write("tip_number_region\n")

## 【API】 YOLO() 加载训练好的检测权重，把模型结构和权重都读进内存，后面直接 predict() 就能用。
print(f"📦 加载检测模型: {WEIGHTS}")
model = YOLO(WEIGHTS)

## 【数据流】 扫描输入目录，收集所有图片路径。sorted 保证顺序一致，结果可复现。
img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
img_paths = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in img_exts])
## 【数据流】 打印图片数量，确认扫描结果。
print(f"📂 找到 {len(img_paths)} 张图片\n")

## 【数据流】 逐张读图。cv2.imread() 读进来是 BGR 格式、numpy 数组。读失败（None）就跳过。
for img_path in img_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ⚠️ 跳过: {img_path.name}")
        continue

    ## 【API】 YOLOv8 推理。输入一张图，输出检测框（xyxy 坐标 + 置信度）。conf=CONF 过滤低分框，imgsz=640 缩放。
    results = model.predict(source=str(img_path), conf=CONF, imgsz=IMGSZ, verbose=False, save=False)

    ## 【数据流】 构造输出 txt 路径，初始化计数器。
    txt_path = OUT_DIR / f"{img_path.stem}.txt" # img_path.stem 是 Path 对象的属性，返回文件名不含扩展名（如 "img_001"）。
    count = 0
    with open(txt_path, "w") as f:
        ## 【API】 results 是 predict() 返回的列表，每个元素对应一张图的检测结果。这里只有 1 个元素。
        for r in results:
            if r.boxes is None:
                continue
            ## 【API】 r.boxes.xywhn 是检测框的归一化 xywh 格式（中心 x、中心 y、宽度、高度，所有值已除以原图宽高归一化到 0-1）。
            ##        r.boxes.conf 是每个框的置信度。
            ## 【为什么】 用 xywhn（归一化中心+宽高）而不是 xyxy，因为直接就是 YOLO txt 要的格式，不用自己再算归一化。
            xywhn = r.boxes.xywhn.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            ## 【为什么】 虽然 predict(conf=CONF) 已经过滤了一次，但这里再过滤一遍是双保险。
            for (x_c, y_c, bw, bh), conf in zip(xywhn, confs):
                if conf < CONF:
                    continue
                ## 【数据流】 写入一行标注。类别 ID 强制写 0（因为只检测一类，不管模型输出什么类别都覆盖为 0）。
                ##          格式和 LabelImg 生成的 YOLO txt 完全一致。
                ## 【为什么】 这是"只检测"模式——不知道框里是什么字符，所以类别全标 0。
                ##          后续由 recognize_only.py 读取这些 txt，填上真正的类别 ID。
                f.write(f"0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
                count += 1

    print(f"  ✅ {img_path.name} → {count} 个框")

## 【数据流】 打印完成信息，显示输出目录的绝对路径。
print(f"\n🎉 检测完成！输出至: {OUT_DIR.resolve()}")