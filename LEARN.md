# 📖 代码学习指南

> 本分支是 `yolov8-bullet-like-tip-ocr` 项目的**逐行注释版**，
> 每个核心 `.py` 文件都加了详细中文注释，适合想搞懂每一行代码的人阅读。

## 推荐阅读顺序

| 步骤 | 文件 | 重点理解 |
|------|------|----------|
| **1** | [`main.py`](main.py) | 端到端 pipeline 总入口，YOLO 检测 + ResNet 识别怎么串起来的 |
| **2** | [`tools/inference/detect_only.py`](tools/inference/detect_only.py) | YOLO 推理输出格式、NMS 后处理 |
| **3** | [`tools/inference/recognize_only.py`](tools/inference/recognize_only.py) | ResNet 分类推理、softmax → class_id |
| **4** | [`tools/data_processing/crop_roi.py`](tools/data_processing/crop_roi.py) | 检测框 → ROI 裁剪 → 预处理 → 分类输入 |
| **5** | [`tools/training/train_resnet18_v2.py`](tools/training/train_resnet18_v2.py) | 数据加载、增强、超参、训练循环 |
| **6** | [`tools/training/train_yolo_v2.py`](tools/training/train_yolo_v2.py) | YOLOv8 微调配置、early stopping |
| **7** | [`eval_end2end.py`](eval_end2end.py) | CAR 计算公式、端到端评估逻辑 |
| **8** | 其余 `tools/` 下的脚本 | 按需阅读 |

## 注释风格说明

每个文件中的注释包含以下标签，方便快速定位信息：

| 标签 | 含义 |
|------|------|
| `【数据流】` | 当前变量的 shape / dtype / 值域 |
| `【为什么】` | 为什么这样写，不这样写会怎样 |
| `【坑】` | 容易踩的坑（API 行为、格式不匹配等） |
| `【API】` | 用到的关键第三方库函数说明 |
| `【TODO】` | 作者还没完全搞懂、待补充的地方 |

## 前置知识

读这些代码前，建议至少了解：

- Python 基础（list/dict/函数/类）
- NumPy 基础（array shape、broadcasting）
- PyTorch 基础（Tensor、nn.Module、DataLoader）
- OpenCV 基础（imread、resize、颜色空间转换）
- YOLO 检测原理（bbox 格式、NMS、mAP）
- 卷积神经网络基本概念（CNN、ResNet、迁移学习）

## 术语速查

| 术语 | 解释 |
|------|------|
| CAR | Character Accuracy Rate，字符正确识别率 |
| mAP50-95 | COCO 标准 mAP，IoU 阈值从 0.5 到 0.95 取平均 |
| ROI | Region of Interest，感兴趣区域（这里指裁剪出的字符区域） |
| CLAHE | 限制对比度自适应直方图均衡化，用于增强低对比度图像 |
| Early Stopping | 验证指标不再提升时提前终止训练，防止过拟合 |

## 数据流总览

```text
原始图片 (BGR, H×W×3)
↓ YOLOv8 检测
检测框 bboxes (N×4, 归一化坐标)
↓ 乘回原图尺寸 + 裁剪
字符 ROI (RGB, 224×224)
↓ ResNet-18 分类
类别 ID (0-35, 对应 0-9/A-Z 去掉 I/O)
↓ 写回 YOLO txt
最终标注文件 (LabelImg 可直接打开)
```

## 如何运行

```bash
# 克隆仓库并切换到注释分支
git clone https://github.com/ZXY-zhu/yolov8-bullet-like-tip-ocr.git
cd yolov8-bullet-like-tip-ocr
git checkout annotated

# 安装依赖
pip install -r requirements.txt

# 下载权重（见 README 4.0 节）

# 直接改 main.py 顶部 4 个路径，然后运行
python main.py
```

> 详细的环境配置和权重下载步骤请参考 [README](README.md)。