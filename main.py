"""
main.py
用途：刻印检测 + 字符识别 → 输出 LabelImg 可读 YOLO txt
说明：
- 输入：图片目录
- 处理：YOLOv8 检测刻印框 → ResNet-18 识别每个框内字符 → 合并为带类别 ID 的 YOLO txt
- 输出：预测标注目录（含 classes.txt + 每张图的 .txt）
- 运行方式：直接改下面 4 个路径，然后 python main.py
"""

## 【数据流】导入所有依赖，cv2读图/裁剪，torch跑模型，PIL做图像格式转换，YOLO是检测模型，transforms是预处理
import cv2
import torch
from pathlib import Path
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

## 【为什么】这 4 个路径是全局配置，改一处就能换数据集/换权重，不用动下面逻辑
# ========== 改这里 ==========
IMAGES_DIR = Path("images")                                # 改为实际图片目录路径
DETECT_WEIGHTS = "weights/best.pt"                         # 改为实际 YOLOv8 检测权重路径
RECOGNIZE_WEIGHTS = "weights/resnet18_best.pth"            # 改为实际 ResNet-18 识别权重路径
OUT_DIR = Path("labels_pred")                              # 改为实际输出标注目录路径
# ==========================

## 【为什么】CONF 太低会出很多误检框，太高会漏检。0.25 是 YOLOv8 的默认推理阈值。
CONF = 0.25    # 置信度阈值：检测框得分低于 0.25 的直接扔掉
IMGSZ = 640    # YOLO 推理时把图片缩到 640×640（保持比例，短边 pad）

## 【为什么】我的 ResNet 只训了 14 类（0-9 是数字，10-13 对应 B/C/D/E）。但 YOLO txt 格式需要 36 类（0-9 + A-Z）。
##         这里做映射：ResNet 输出的 10 对应字符 11（因为跳过了 10=A），11 对应 12（跳过了 A），以此类推。
## 【坑】注意 10 没有对应 A（数据里没有 A 这个类），所以映射时跳过了 10。
RESNET_14_TO_36 = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
    7: 7, 8: 8, 9: 9, 10: 11, 11: 12, 12: 13, 13: 14,
}

## 【数据流】生成 36 个类名：["0","1",...,"9","A","B",...,"Z"]，写进 classes.txt 供 LabelImg 读取。
CLASS_NAMES_36 = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]


## 【API】YOLO() 加载训练好的检测权重。这一步把模型结构和权重都读进内存，后面直接 predict() 就能用。
print(f"📦 加载检测模型: {DETECT_WEIGHTS}")
detect_model = YOLO(DETECT_WEIGHTS)

## 【数据流】从.pth文件加载训练好的ResNet权重字典
##         map_location="cpu"意思是先加载到CPU（后面再.to(device)移到GPU）
print(f"📦 加载识别模型: {RECOGNIZE_WEIGHTS}")
raw_ckpt = torch.load(RECOGNIZE_WEIGHTS, map_location="cpu")
## 【坑】如果权重是用别的训练代码保存的，key 前面可能带了 backbone. 前缀（比如 backbone.conv1.weight），
##      但 models.resnet18() 期望的 key 是 conv1.weight。这行检查并去掉前缀，防止 load_state_dict 报错。
if any(k.startswith("backbone.") for k in raw_ckpt.keys()):
    raw_ckpt = {k.replace("backbone.", ""): v for k, v in raw_ckpt.items()}

## 【为什么】weights=None 表示不加载 ImageNet 预训练权重（因为要加载自己训好的）。只创建 ResNet-18 的空壳结构。
recognize_model = models.resnet18(weights=None)
## 【数据流】替换最后一层全连接层，从 1000 类改成 14 类。和训练时一模一样的结构。
recognize_model.fc = torch.nn.Linear(recognize_model.fc.in_features, 14)
## 【API】把权重字典加载进模型。现在模型结构和权重都就位了，可以推理。
recognize_model.load_state_dict(raw_ckpt)
## 【为什么】切换为评估模式。关掉 Dropout、BN 用 running 统计。推理时必须写，否则结果不稳定。
recognize_model.eval()

## 【数据流】如果有 GPU 就把模型搬到 GPU 上，推理更快。没有就用 CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
recognize_model.to(device)

## 【为什么】最后一步 Normalize 是 ImageNet 数据集的 RGB 均值和标准差，模型预训练时用了这组值，推理时保持一致。
# transforms.Compose 把这个流水线存成一个变量 transform，后面每次来一张新图，写一行 transform(roi_rgb) 就全做完了。
transform = transforms.Compose([
    transforms.Resize((224, 224)), # ResNet-18只可识别224 * 224的图片
    transforms.ToTensor(), # 这步把 0-255 压缩到 0.0-1.0，同时把形状从 (高,宽,通道) 翻转为 (通道,高,宽)——PyTorch 要求的格式。
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 把每个像素值再"平移缩放"一下。公式是 (像素值 - mean) / std。做完之后，大部分值落在 -2 到 +2 之间。
])

## 【数据流】创建输出目录，写 classes.txt。LabelImg 打开标注时需要这个文件知道类别名。
OUT_DIR.mkdir(parents=True, exist_ok=True) # 父目录不存在时递归创建
classes_path = OUT_DIR / "classes.txt"
with open(classes_path, "w", encoding="utf-8") as f:
    f.write("\n".join(CLASS_NAMES_36) + "\n")
print(f"📄 已生成 classes.txt ({len(CLASS_NAMES_36)} 类)")

## 【数据流】扫描输入目录，收集所有图片路径。sorted 保证顺序一致，结果可复现。
img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# IMAGES_DIR.iterdir() 是返回目录下所有条目的迭代器。p.suffix 返回扩展名（含点号）。sorted 保证每次运行结果顺序一致，可复现。
img_paths = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in img_exts])
print(f"📂 找到 {len(img_paths)} 张图片，开始检测+识别...\n")

## 【数据流】逐张读图。cv2 读进来是 BGR 格式、numpy 数组。读失败就跳过。
for img_path in img_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ⚠️ 跳过（读图失败）: {img_path.name}")
        continue

    ## 【为什么】记下原图高宽，后面要把检测框坐标归一化（除以 w 和 h），因为 YOLO txt 格式要求归一化坐标。
    h, w = img.shape[:2]
    ## 【API】YOLOv8 推理。输入一张图，输出检测框（xyxy 坐标 + 置信度）。conf=CONF 过滤低分框，imgsz=640 缩放。
    results = detect_model.predict(
        source=str(img_path), conf=CONF, imgsz=IMGSZ, verbose=False, save=False,
    )

    txt_path = OUT_DIR / f"{img_path.stem}.txt" # img_path.stem 是 Path 对象的属性，返回文件名不含扩展名。
    count = 0
    with open(txt_path, "w", encoding="utf-8") as f:
        ## 【API】r.boxes.xyxy 是检测框的左上右下坐标（tensor 格式，在 GPU 上）。
        ##       .cpu().numpy() 搬到 CPU 并转成 numpy 数组方便后面算
        for r in results: # result 一次输入几个就有几个元素，每个元素包含这张图的所有检测框信息，这里 result 只有一个元素。
            if r.boxes is None: # 无框时跳过
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            ## 【为什么】虽然 YOLO 的 predict 已经用 conf 过滤了一次，但这里再过滤一遍是双保险。
            for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                if conf < CONF:
                    continue
                x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2]) # 把浮点坐标转成整数像素坐标。
                ## 【数据流】用检测框坐标从原图裁剪出 ROI（Region of Interest，就是框里的字符区域）。
                roi = img[y1_i:y2_i, x1_i:x2_i]
                ## 【为什么】检查框是否合法（比如坐标超出图片边界时裁剪出来是空数组）。
                if roi.size == 0: # roi.size 是 numpy 数组的属性，返回元素总数（高×宽×通道）。
                    continue

                ## 【坑】cv2 读的是 BGR，但 ResNet/PIL 期望 RGB。先 BGR→RGB 再转 PIL Image。
                ##     .convert("RGB") 确保是 3 通道（防止灰度图出错）。
                roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                roi_rgb = roi_pil.convert("RGB") # 强制将 roi_pil 改为 3 通道，确保图片是 3 通道 RGB。
                ## 【数据流】预处理三步（Resize→ToTensor→Normalize），
                ##         然后 unsqueeze(0) 在第 0 维加一个 batch 维度（模型期望输入是 [batch, 3, 224, 224]，
                ##         单张图就是 [1, 3, 224, 224]），最后搬到 GPU/CPU。
                roi_tensor = transform(roi_rgb).unsqueeze(0).to(device) # .unsqueeze() 在指定位置"加一个大小为 1 的维度"

                ## 【数据流】推理模式，不建计算图。
                ##         模型输出 [1, 14] 的 logits，argmax(dim=1) 取最大值的索引（0-13），.item() 转成 Python 整数。
                ## 【为什么】torch.no_grad() 省显存。这里不需要梯度，因为不训练。
                with torch.no_grad():
                    pred_id = recognize_model(roi_tensor).argmax(dim=1).item() # .argmax(dim=1) 在第 1 维上找最大值的位置。.item() 把只有一个元素的张量提取成 Python 数字。

                ## 【数据流】把 ResNet 的 14 类 ID 映射回 36 类 ID，对应 classes.txt 里的索引。
                class_id = RESNET_14_TO_36.get(pred_id, 0)
                ## 【为什么】YOLO txt 格式要求：每行 class_id x_center y_center width height，全部除以原图宽高归一化到 0-1 之间。
                x_c = ((x1 + x2) / 2) / w
                y_c = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                ## 【数据流】写入一行标注。格式和 LabelImg 生成的完全一致，可以直接用 LabelImg 打开验证
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
                count += 1

    print(f"  ✅ {img_path.name} → {count} 个框")

print(f"\n🎉 完成！")
print(f"   标签目录: {OUT_DIR.resolve()}")
print(f"   classes:  {classes_path}")