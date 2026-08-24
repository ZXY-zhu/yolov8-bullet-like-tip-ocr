"""
main.py — 刻印检测 + 字符识别 → 输出 LabelImg 可读 YOLO txt（最终交付）

输入：图片目录
输出：每个图片对应一个 .txt（框位置 + 识别字符类别），自动生成 classes.txt

用法：
    C:\\Users\\BeLig\\miniconda3\\envs\\yolov8\\python.exe main.py ^
        --images <图片目录> ^
        --detect-weights <YOLO权重.pt> ^
        --recognize-weights <ResNet权重.pth> ^
        --out <输出txt目录>
"""

import argparse
from pathlib import Path
import cv2
import torch
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO


# ============ 类别映射 ============
# ResNet 13 类 → 36 类 YOLO 格式
RESNET_13_TO_36 = {
    0: 0,   # 0
    1: 1,   # 1
    2: 2,   # 2
    3: 3,   # 3
    4: 4,   # 4
    5: 5,   # 5
    6: 7,   # 7（训练数据无字符 6）
    7: 8,   # 8
    8: 9,   # 9
    9: 11,  # B（训练数据无字符 A）
    10: 12, # C
    11: 13, # D
    12: 14, # E
}

# 36 类 ID → 字符名（写 classes.txt 用）
CLASS_NAMES_36 = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]


def main():
    parser = argparse.ArgumentParser(description="刻印检测+识别 → 输出 LabelImg 可读 YOLO txt")
    parser.add_argument("--images", required=True, help="原始图片目录")
    parser.add_argument("--detect-weights", required=True, help="YOLO 检测权重 .pt")
    parser.add_argument("--recognize-weights", required=True, help="ResNet 分类权重 .pth")
    parser.add_argument("--out", required=True, help="输出 txt 目录")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO 置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理尺寸")
    args = parser.parse_args()

    # ---- 加载 YOLO ----
    print(f"📦 加载检测模型: {args.detect_weights}")
    detect_model = YOLO(args.detect_weights)

    # ---- 加载 ResNet ----
    print(f"📦 加载识别模型: {args.recognize_weights}")
    raw_ckpt = torch.load(args.recognize_weights, map_location="cpu")
    if any(k.startswith("backbone.") for k in raw_ckpt.keys()):
        raw_ckpt = {k.replace("backbone.", ""): v for k, v in raw_ckpt.items()}

    recognize_model = models.resnet18(weights=None)
    recognize_model.fc = torch.nn.Linear(recognize_model.fc.in_features, 13)
    recognize_model.load_state_dict(raw_ckpt)
    recognize_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recognize_model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ---- 创建输出目录 + 自动生成 classes.txt ----
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes_path = out_dir / "classes.txt"
    with open(classes_path, "w", encoding="utf-8") as f:
        f.write("\n".join(CLASS_NAMES_36) + "\n")
    print(f"📄 已生成 classes.txt ({len(CLASS_NAMES_36)} 类)")

    # ---- 处理图片 ----
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_paths = sorted([p for p in Path(args.images).iterdir() if p.suffix.lower() in img_exts])
    print(f"📂 找到 {len(img_paths)} 张图片，开始检测+识别...\n")

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️ 跳过（读图失败）: {img_path.name}")
            continue

        h, w = img.shape[:2]

        # YOLO 检测
        results = detect_model.predict(
            source=str(img_path),
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
            save=False,
        )

        txt_path = out_dir / f"{img_path.stem}.txt"
        count = 0

        with open(txt_path, "w", encoding="utf-8") as f:
            for r in results:
                if r.boxes is None:
                    continue

                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()

                for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                    if conf < args.conf:
                        continue

                    # 裁 ROI
                    x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
                    roi = img[y1_i:y2_i, x1_i:x2_i]
                    if roi.size == 0:
                        continue

                    # 预处理 → ResNet 推理
                    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    roi_gray = roi_pil.convert("L")
                    roi_rgb = Image.merge("RGB", [roi_gray] * 3)
                    roi_tensor = transform(roi_rgb).unsqueeze(0).to(device)

                    with torch.no_grad():
                        pred_id = recognize_model(roi_tensor).argmax(dim=1).item()

                    # 13 类 → 36 类
                    class_id = RESNET_13_TO_36.get(pred_id, 0)

                    # 归一化 xywh
                    x_c = ((x1 + x2) / 2) / w
                    y_c = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                    f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
                    count += 1

        print(f"  ✅ {img_path.name} → {count} 个框")

    print(f"\n🎉 完成！")
    print(f"   标签目录: {out_dir.resolve()}")
    print(f"   classes:  {classes_path}")
    print(f"\n📌 用 LabelImg 打开：")
    print(f'   labelImg "{args.images}" "{out_dir}" "{classes_path}"')


if __name__ == "__main__":
    main()