"""
main.py
用途：刻印检测 + 字符识别 → 输出 LabelImg 可读 YOLO txt
说明：
- 输入：图片目录
- 处理：YOLOv8 检测刻印框 → ResNet-18 识别每个框内字符 → 合并为带类别 ID 的 YOLO txt
- 输出：预测标注目录（含 classes.txt + 每张图的 .txt）
- 运行方式：直接改下面 4 个路径，然后 python main.py
"""

import cv2
import torch
from pathlib import Path
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

# ========== 改这里 ==========
IMAGES_DIR = Path("images")                                # 改为实际图片目录路径
DETECT_WEIGHTS = "weights/best.pt"                         # 改为实际 YOLOv8 检测权重路径
RECOGNIZE_WEIGHTS = "weights/resnet18_best.pth"            # 改为实际 ResNet-18 识别权重路径
OUT_DIR = Path("labels_pred")                              # 改为实际输出标注目录路径
# ==========================

CONF = 0.25
IMGSZ = 640

RESNET_14_TO_36 = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
    7: 7, 8: 8, 9: 9, 10: 11, 11: 12, 12: 13, 13: 14,
}

CLASS_NAMES_36 = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]

print(f"📦 加载检测模型: {DETECT_WEIGHTS}")
detect_model = YOLO(DETECT_WEIGHTS)

print(f"📦 加载识别模型: {RECOGNIZE_WEIGHTS}")
raw_ckpt = torch.load(RECOGNIZE_WEIGHTS, map_location="cpu")
if any(k.startswith("backbone.") for k in raw_ckpt.keys()):
    raw_ckpt = {k.replace("backbone.", ""): v for k, v in raw_ckpt.items()}

recognize_model = models.resnet18(weights=None)
recognize_model.fc = torch.nn.Linear(recognize_model.fc.in_features, 14)
recognize_model.load_state_dict(raw_ckpt)
recognize_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
recognize_model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

OUT_DIR.mkdir(parents=True, exist_ok=True)
classes_path = OUT_DIR / "classes.txt"
with open(classes_path, "w", encoding="utf-8") as f:
    f.write("\n".join(CLASS_NAMES_36) + "\n")
print(f"📄 已生成 classes.txt ({len(CLASS_NAMES_36)} 类)")

img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
img_paths = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in img_exts])
print(f"📂 找到 {len(img_paths)} 张图片，开始检测+识别...\n")

for img_path in img_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ⚠️ 跳过（读图失败）: {img_path.name}")
        continue

    h, w = img.shape[:2]
    results = detect_model.predict(
        source=str(img_path), conf=CONF, imgsz=IMGSZ, verbose=False, save=False,
    )

    txt_path = OUT_DIR / f"{img_path.stem}.txt"
    count = 0
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                if conf < CONF:
                    continue
                x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
                roi = img[y1_i:y2_i, x1_i:x2_i]
                if roi.size == 0:
                    continue

                roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                roi_rgb = roi_pil.convert("RGB")
                roi_tensor = transform(roi_rgb).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred_id = recognize_model(roi_tensor).argmax(dim=1).item()

                class_id = RESNET_14_TO_36.get(pred_id, 0)
                x_c = ((x1 + x2) / 2) / w
                y_c = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
                count += 1

    print(f"  ✅ {img_path.name} → {count} 个框")

print(f"\n🎉 完成！")
print(f"   标签目录: {OUT_DIR.resolve()}")
print(f"   classes:  {classes_path}")