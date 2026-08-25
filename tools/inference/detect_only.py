"""
tools/inference/detect_only.py
用途：只跑 YOLO 检测，输出框（类别全 0）
说明：
- 输入：待检测图片目录
- 输出：YOLO 格式 txt 标注（所有类别 ID 强制写为 0）+ classes.txt
- 运行方式：直接改下面 3 个路径，然后 python tools/detect_only.py
"""

from pathlib import Path
import cv2
from ultralytics import YOLO

# ========== 改这里 ==========
IMAGES_DIR = Path("images")                                    # 改为实际待检测图片目录路径
WEIGHTS = "weights/best.pt"                                    # 改为实际模型权重路径
OUT_DIR = Path("labels_pred")                                  # 改为实际输出目录路径
CONF = 0.25
IMGSZ = 640
# ============================

OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUT_DIR / "classes.txt", "w") as f:
    f.write("tip_number_region\n")

print(f"📦 加载检测模型: {WEIGHTS}")
model = YOLO(WEIGHTS)

img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
img_paths = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in img_exts])
print(f"📂 找到 {len(img_paths)} 张图片\n")

for img_path in img_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ⚠️ 跳过: {img_path.name}")
        continue

    results = model.predict(source=str(img_path), conf=CONF, imgsz=IMGSZ, verbose=False, save=False)

    txt_path = OUT_DIR / f"{img_path.stem}.txt"
    count = 0
    with open(txt_path, "w") as f:
        for r in results:
            if r.boxes is None:
                continue
            xywhn = r.boxes.xywhn.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for (x_c, y_c, bw, bh), conf in zip(xywhn, confs):
                if conf < CONF:
                    continue
                f.write(f"0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
                count += 1

    print(f"  ✅ {img_path.name} → {count} 个框")

print(f"\n🎉 检测完成！输出至: {OUT_DIR.resolve()}")