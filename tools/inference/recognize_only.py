"""
tools/inference/recognize_only.py
用途：读纯框 YOLO txt 裁剪 ROI → ResNet 识别 → 输出带类别 ID 的 YOLO txt
说明：
- 输入：图片目录 + 类别全 0 的纯框 YOLO txt 目录
- 输出：带识别类别 ID 的 YOLO txt 目录
- 运行方式：直接改下面 4 个路径，然后 python tools/recognize_only.py
"""

from pathlib import Path
import cv2
import torch
from torchvision import models, transforms
from PIL import Image

# ========== 改这里 ==========
IMAGES_DIR = Path("images")                                          # 改为实际图片目录路径
LABELS_DIR = Path("labels_for_recognize")                            # 改为实际纯框 YOLO txt 目录路径
WEIGHTS = "weights/resnet18_char_classifier_best.pth"                # 改为实际 ResNet 权重路径
OUT_DIR = Path("recognize_pred_yolo")                                # 改为实际输出目录路径
# ============================

# 13 类 → 36 类
RESNET_13_TO_36 = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
    6: 7, 7: 8, 8: 9, 9: 11, 10: 12, 11: 13, 12: 14
}

print(f"📦 加载识别模型: {WEIGHTS}")
raw_ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
if any(k.startswith("backbone.") for k in raw_ckpt.keys()):
    raw_ckpt = {k.replace("backbone.", ""): v for k, v in raw_ckpt.items()}

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 13)
model.load_state_dict(raw_ckpt)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

OUT_DIR.mkdir(parents=True, exist_ok=True)

img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
img_paths = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in img_exts])

print(f"📂 处理 {len(img_paths)} 张图片...\n")

for img_path in img_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]
    label_path = LABELS_DIR / f"{img_path.stem}.txt"
    if not label_path.exists():
        continue

    txt_path = OUT_DIR / f"{img_path.stem}.txt"
    with open(label_path, "r") as f_in, open(txt_path, "w") as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            _, x_c, y_c, bw, bh = parts[:5]
            x_c, y_c, bw, bh = map(float, [x_c, y_c, bw, bh])

            # 裁 ROI 给 ResNet
            abs_x1 = int((x_c - bw / 2) * w)
            abs_y1 = int((y_c - bh / 2) * h)
            abs_x2 = int((x_c + bw / 2) * w)
            abs_y2 = int((y_c + bh / 2) * h)

            roi = img[abs_y1:abs_y2, abs_x1:abs_x2]
            if roi.size == 0:
                # 框无效，写默认类 0，坐标照写
                f_out.write(f"0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
                continue

            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            roi_gray = roi_pil.convert("L")
            roi_rgb = Image.merge("RGB", [roi_gray] * 3)
            roi_tensor = transform(roi_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_id = model(roi_tensor).argmax(dim=1).item()

            mapped_id = RESNET_13_TO_36.get(pred_id, 0)

            f_out.write(f"{mapped_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"  ✅ {img_path.name}")

print(f"\n🎉 识别完成！输出至: {OUT_DIR.resolve()}")