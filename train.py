from ultralytics import YOLO

# 1. 创建一个 YOLOv8 模型（用别人练过的“半成品”）
model = YOLO("yolov8n.pt")

# 2. 开始训练
model.train(
    data="C:/Users/BeLig/Desktop/ocr/no/data.yaml.local",  # 刚才的说明书
    epochs=5,          # 先只训 5 次，试试水
    imgsz=640,         # 图片缩放大小
    batch=4,           # 一次看 4 张图（CPU 不吃力）
    device="cpu"       # 用 CPU
)

### 数据清洗说明
#训练初期发现单张图像（20260719_173736_876_2.jpg）存在标注异常（多类别框重叠/重复）。
#已对原始标注文件（`labels_raw/`）及 YOLO 格式标签（`labels/`）进行人工修正，确保单目标单框。
#受限于本地 CPU 算力，未对模型进行重新训练，当前权重基于修正前数据分布。
#后续将在 GPU 环境下进行完整重训以验证数据修正效果。

"""
Validating C:\\Users\BeLig\PycharmProjects\YOLOv8Project\runs\detect\train-3\weights\best.pt...
Ultralytics 8.4.115  Python-3.10.20 torch-2.13.0+cpu CPU (AMD Ryzen 7 7730U with Radeon Graphics)
Model summary (fused): 73 layers, 3,005,843 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 227/227 1.3s/it 5:06
                   all       1813       6231      0.979      0.986      0.994      0.889
Speed: 3.5ms preprocess, 126.8ms inference, 0.0ms loss, 0.8ms postprocess per image
Results saved to C:\\Users\BeLig\PycharmProjects\YOLOv8Project\runs\detect\train-3

进程已结束，退出代码为 0
"""