# YOLOv8 类子弹头状器件刻印区域检测

[![Status](https://img.shields.io/badge/Status-Algorithm%20Prototype%20✅-green.svg)](https://github.com/ZXY-zhu/yolov8-bullet-like-tip-ocr#七当前进度说明)
[![CNN Val Acc](https://img.shields.io/badge/CNN%20Val%20Acc-95.23%25-brightgreen.svg)](https://github.com/ZXY-zhu/yolov8-bullet-like-tip-ocr#53-定制-cnn-字符分类)
[![ResNet-18 Val Acc](https://img.shields.io/badge/ResNet--18%20Val%20Acc-99.19%25-success.svg)](https://github.com/ZXY-zhu/yolov8-bullet-like-tip-ocr#54-resnet-18-迁移学习)
[![YOLOv8 mAP50-95](https://img.shields.io/badge/YOLOv8%20mAP50--95-0.921-blue.svg)](https://github.com/ZXY-zhu/yolov8-bullet-like-tip-ocr#51-yolov8-检测模型)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg)](https://docs.ultralytics.com/)

## 目录

- [一、项目背景](#一项目背景)
- [二、数据集说明](#二数据集说明)
- [三、环境依赖](#三环境依赖)
- [四、快速开始](#四快速开始)
  - [4.0 使用预训练权重（推荐）](#40-使用预训练权重推荐最快上手)
  - [4.1 准备数据配置](#41-准备数据配置)
  - [4.2 训练模型](#42-训练模型)
  - [4.3 推理测试](#43-推理测试)
- [五、实验结果](#五实验结果)
- [六、项目结构](#六项目结构)
- [七、当前进度](#七当前进度说明)
- [八、后续工作](#八后续工作)
- [九、更新日志](#九更新日志)
- [⚠️ 关于 CPU / GPU 的说明](#️-关于-cpu--gpu-的说明)

> [!TIP]
> **📖 带详细注释的代码版本**（建设中）
> 
> 如果你想逐行学习每个脚本的实现细节，请切换到 [`annotated`](../../tree/annotated) 分支，
> 每个核心 `.py` 文件都有逐行中文注释，适合入门学习。

## 一、项目背景

在工业质检场景中，需要在金属器件尖端表面识别凹印/凸印编号。该类目标存在以下难点：

- 金属反光强，局部过曝
- 刻印字符尺寸小、对比度低
- 部分样本无刻印（负样本）
- 图像存在倾斜、旋转与尺度变化

> 注：本项目为个人验证性质，基于 1813 张旧数据 + 536 张新数据完成算法原型验证，不涉及真实产线部署。

## 二、数据集说明

- 来源：实习单位内部采集（受保密协议限制，不公开原始数据）
- 规模：旧数据 1813 张 + 新数据 536 张（混合训练集）
- 标注：YOLO 格式，单类别（刻印区域）
- 划分：训练集 1631 张，验证集 182 张
- 特殊处理：对无刻印样本引入空标签文件，避免训练阶段漏检惩罚

### 标注预处理逻辑

本项目为单类别检测任务，仅关注“刻印区域”。然而在早期标注及导出过程中，部分 YOLO 标签文件的首字段（类别 ID）可能为数字、字母或非零映射结果，而非统一的 `0`。

YOLOv8 在解析标签时，若类别 ID 无法映射到 `data.yaml` 中定义的类别索引，会**静默忽略整行标注**，既不报错也不告警。这可能导致一种隐蔽问题：**标签文件存在，但训练时并未真正生效**。

为避免上述问题，同时保证 OCR 输入纯净，采取以下策略：

- 标注阶段仅框选印刷体刻印区域；
- 手写体编号视为背景，不予标注，避免污染后续 OCR 输入；
- 编写 `tools/labelsTOyolo.py`，对标签进行标准化处理：
  - 不修改原始标签文件；
  - 读取 `labels/*.txt`；
  - 将每行第一个字段统一改写为 `"0"`，以适配单类别检测任务；
  - 保留剩余四个字段（归一化后的 x, y, w, h）；
  - 输出至 `labels_yolo/*.txt`。

该方案确保：
- 检测阶段仅关注“刻印区域是否存在”；
- 字符内容由后续 OCR 模块独立识别；
- 消除因类别 ID 不一致导致的标注静默失效问题；
- 单类别训练标签保持统一、可控。

## 三、环境依赖

安装命令：
```bash
pip install -r requirements.txt
```

> 建议使用 GPU 环境以获得正常训练速度与性能。CPU 环境仅适合流程验证。

主要依赖：

- Python>=3.8
- PyTorch>=2.0
- Ultralytics YOLOv8
- OpenCV
- paddlepaddle==3.3.1
- paddleocr==3.7.0
- torchvision
- scikit-learn
- Matplotlib

## 四、快速开始

### 4.0 使用预训练权重（推荐，最快上手）

#### 1. 克隆 & 安装

```bash
git clone https://github.com/ZXY-zhu/yolov8-bullet-like-tip-ocr.git
cd yolov8-bullet-like-tip-ocr
pip install -r requirements.txt
```

#### 2. 下载权重

从 [Releases v1.0.0](https://github.com/ZXY-zhu/yolov8-bullet-like-tip-ocr/releases/tag/v1.0.0) 下载以下文件，放入 `weights/`：

```bash
mkdir -p weights
curl -L -o weights/best.pt https://github.com/ZXY-zhu/yolov8-bullet-like-tip-ocr/releases/download/v1.0.0/best.pt
curl -L -o weights/resnet18_best.pth https://github.com/ZXY-zhu/yolov8-bullet-like-tip-ocr/releases/download/v1.0.0/resnet18_best.pth
```

#### 3. 运行端到端推理

```bash
#单张图片
python main.py --input path/to/image.jpg --output results/

#批量推理
python main.py --input path/to/images/ --output results/

#端到端评估（需准备标注数据）
python eval_end2end.py
```

> YOLOv8n 预训练权重首次运行时会由 Ultralytics 自动下载，无需手动准备。

### 4.1 准备数据配置

复制示例配置，并根据实际路径修改，生成本地私有配置文件：

#### Linux / Mac
```bash
cp data.yaml.example data.yaml
```

#### Windows PowerShell
```bash
copy data.yaml.example data.yaml
```

> `data.yaml.example` 为配置模板，`data.yaml` 为实际使用的本地配置文件。  
> `data.yaml` 已被加入 `.gitignore`，不会上传至 GitHub，避免泄露本地或服务器路径。

数据配置内容（服务器环境示例）：

```yaml
train: /root/autodl-tmp/dataset/train.txt
val: /root/autodl-tmp/dataset/val.txt
names:
  0: tip_number_region
```

### 4.2 训练模型

#### GPU（推荐）
```bash
yolo train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640 batch=8 device=0
```

#### CPU（仅调试）
```bash
yolo train model=yolov8n.pt data=data.yaml epochs=5 imgsz=640 batch=2 device=cpu
```

### 4.3 推理测试
```bash
yolo predict model=runs/detect/bullet_tip_v1/weights/best.pt source=dataset/images/val save=True
```

> 注：`bullet_tip_v1` 为示例训练输出目录，实际使用时请根据 `runs/detect/` 下的具体文件夹名调整。

## 五、实验结果

### 5.1 YOLOv8 检测模型

| 指标 | 旧数据（1813 张） | 新旧混合微调（8.25） |
|---|---|---|
| mAP50 | 0.994 | 0.994 |
| mAP50-95 | 0.889 | **0.921** |
| Precision | 0.979 | 0.979 |
| Recall | 0.986 | 0.987 |
| 训练设备 | GPU AutoDL | GPU AutoDL |
| Epochs | 50 | 16（early stopping） |

### 5.1.1 训练曲线

**旧模型（1813 张，50 Epochs）：**

![YOLOv8 旧模型训练曲线](https://raw.githubusercontent.com/ZXY-zhu/yolov8-bullet-like-tip-ocr/main/assets/results_v1.png)

**新模型（新旧混合微调，16 Epochs early stopping）：**

![YOLOv8 v2 训练曲线](https://raw.githubusercontent.com/ZXY-zhu/yolov8-bullet-like-tip-ocr/main/assets/results_v2.png)

> **注**：新模型（v2）因采用 Early Stopping（16 Epochs）且包含跨域混合数据，训练曲线初期存在一定震荡，但 mAP50-95 提升至 0.921，验证了混合数据对精度的增益。

### 5.2 通用 PaddleOCR 基线（CAR 天花板）

| 方案 | CAR | 说明 |
|---|---|---|
| 修正后基线（带过滤） | 5.54% | 评估脚本 bug 修正后的真实值 |
| v2 预处理全量（1813 张） | **8.54%** | CLAHE + 实心化 + 48px + 320 padding，通用 OCR 天花板 |
| 4 位完全匹配率 | 0% | 主要错误：OCR 输出 `?`（DB 检测返回空） |

### 5.3 定制 CNN 字符分类（旧数据 6189 张，13 类）

| 实验 | 最佳 Val Acc | 最终 Train Acc | 最终 Val Loss |
|---|---|---|---|
| CNN Baseline 50ep | 94.43% | 97.48% | 0.3254 |
| CNN Baseline 100ep | 95.23% | 98.97% | 0.4655 |
| + Random Erase 100ep | **95.23%** | 97.92% | **0.3716** |

### 5.4 ResNet-18 迁移学习

| 版本 | 数据 | 输入 | 最佳 Val Acc | 备注 |
|---|---|---|---|---|
| v1 ⚠️ | 旧数据 6189 张，13 类 | 灰度 48→224 | 98.63% | 输入 pipeline 有缺陷（上采样） |
| **v2 ✅** | **混合 8067 张，14 类** | **RGB 224×224** | **99.19%** | **补齐 6/E，SOTA** |

### 5.5 端到端识别评估对比

| 指标 | 旧模型（536 张） | 新模型（2212 张） | 变化 |
|---|---|---|---|
| CAR（仅字符） | 81.48% | **97.23%** | +15.75% |
| CAR（含0框匹配） | — | **97.23%** | — |
| 完全匹配率 | 49.63% | **79.11%** | +29.48% |
| 0框正确跳过 | — | 7 张 | — |
| 6 准确率 | 0% | **100%** | ✅ |
| E 准确率 | 0% | **96.05%** | ✅ |
| B 准确率 | 29.20% | **93.56%** | +64.36% |

> 新模型：YOLOv8 混合微调 + ResNet-18 v2（RGB），全自动端到端推理 2212 张。
> 旧模型：YOLOv8 旧权重 + ResNet-18 v1，人工修正框 536 张。
> 剩余混淆集中在 0↔1/4/C/D（形状相似，属数据物理限制）。

### 5.6 关键对比总览

| 方案 | 字符识别准确率 | 提升倍数（vs 通用 OCR） |
|---|---|---|
| 通用 PaddleOCR（CAR） | 8.54% | 1× |
| 定制 CNN（最终版） | 95.23% | **11.2×** |
| ResNet-18 v1 ⚠️ | 98.63% | **11.5×** |
| ResNet-18 v2 ✅ | **99.19%** | **11.6×** |

### 5.7 预处理 A/B 测试与选型

为突破通用 PaddleOCR 在工业刻印 ROI 上的识别瓶颈，对三个预处理方案进行了 100 张图像的 A/B 测试：

- **v2（CLAHE + 9×9 模糊 + 3×3 实心化 + 48px 归一化 + 320 Padding）**：CAR = 13.35%（最优）
- **Plan A（基线 + 2×2 膨胀，无归一化）**：CAR = 9.94%
- **Plan C（5×5 模糊 + 开运算 + 64px）**：CAR = 5.68%

基于 A/B 测试结果，选定 **v2 管道** 进行 1813 张全量测试，最终 CAR = **8.54%**。确认此为通用 PaddleOCR 在当前数据集上的真实天花板。

## 六、项目结构
```text
├── assets/                    # 静态资源（README 用图）
│   ├── results_v1.png         # 旧模型训练曲线（50 Epochs）
│   └── results_v2.png         # 新模型训练曲线（16 Epochs, early stopping）
├── docs/                      # 项目文档
│   └── experiment_log.md      # 完整实验记录（从 YOLO 到 CNN 到 ResNet 的演进）
├── learn/                     # 代码学习脚本（不提交）
├── weekly_reports/            # 个人实习周报（不提交）
├── notes.md                   # 个人学习笔记（不提交）
├── requirements.txt           # 项目依赖清单
├── data.yaml.example          # 数据配置模板（参考用）
├── data.yaml                  # 本地私有配置（不提交，由 data.yaml.example 复制生成）
├── .gitignore                 # Git 忽略规则配置
├── eval_end2end.py            # 端到端识别准确率评估（按 y 排序竖排比对）
├── main.py                    # 刻印检测+识别端到端推理脚本（YOLO + ResNet → YOLO txt）
├── tools/
│   ├── archive/               # 历史调试脚本归档（不提交）
│   ├── data_processing/       # 数据预处理脚本
│   │   ├── crop_roi.py          # YOLO 检测 + ROI 裁剪
│   │   ├── crop_roi_v2.py       # ROI 裁剪 v2（优化预处理）
│   │   └── labelsTOyolo.py      # 标签标准化（类别 ID 统一为 0）
│   ├── inference/             # 推理与评估脚本
│   │   ├── detect_only.py       # 只跑 YOLO 检测，输出类别全 0 的 txt
│   │   ├── eval_detection.py    # 基于 IoU 的检测评估（Recall/Precision/F1）
│   │   ├── eval_recognition.py  # 识别准确率评估（CAR/完全匹配/混淆矩阵）
│   │   ├── predict_batch.py     # 批量 PaddleOCR 识别脚本
│   │   └── recognize_only.py    # 纯框 YOLO txt → ResNet 识别 → 带类别 YOLO txt
│   ├── outputs/               # 批量识别 CSV 输出（不提交）
│   ├── tests/                 # 测试与验证脚本
│   │   ├── crop_cnn_dataset.py  # CNN 数据集制作（灰度 48×48）
│   │   ├── test_env.py          # 环境测试脚本
│   │   └── test_single_image.py # PaddleOCR 单图推理测试脚本
│   ├── training/              # 训练脚本
│   │   ├── train.py             # YOLOv8 训练（CPU 调试用）
│   │   ├── train_yolo_v2.py     # YOLOv8 混合数据微调（AutoDL）
│   │   ├── train_cnn_aug.py     # 定制 CNN（+ Random Erase, Val Acc 95.23%）
│   │   ├── train_resnet18.py    # ResNet-18 v1（灰度 48→224, Val Acc 98.63% ⚠️）
│   │   └── train_resnet18_v2.py # ResNet-18 v2（RGB, 14类, Val Acc 99.19% ✅）
│   └── evaluate_accuracy.py   # 标准 OCR 评估（CAR/CER + 位置准确率）
├── weights/                   # 训练好的模型权重（不提交）
│   ├── best.pt                # YOLOv8 最佳检测权重
│   └── resnet18_best.pth      # ResNet-18 最佳分类权重
├── runs/                      # 训练输出目录（不提交）
├── yolov8n.pt                 # YOLOv8n 官方预训练权重（不提交，首次自动下载）
├── LEARN.md                   # 代码学习指南（annotated 分支专属）
└── README.md
```

说明：
- 带（不提交）标记的文件/目录已加入 `.gitignore`
- `data.yaml` 仅存在于本地，由使用者根据 `data.yaml.example` 自行创建
- 所有工具脚本中的路径已脱敏为示例相对路径，使用时改为实际路径即可

## 七、当前进度说明

### ✅ 已完成
- 刻印区域检测模型训练完成（旧数据 mAP50 0.994 / 混合微调 mAP50-95 **0.921**）
- PaddleOCR 3.x 环境验证，确认通用 OCR 天花板 CAR 8.54%
- YOLO 检测 → ROI 裁剪 → OCR/分类 端到端链路打通
- CNN 训练数据集制作完成：6189 张灰度 48×48，13 类
- 定制 CNN 训练完成（100ep + Random Erase），Val Acc **95.23%**
- ResNet-18 v2 混合数据重训完成（RGB 224×224，14 类），Val Acc **99.19%** ✅
- 新模型端到端评估 2212 张：CAR **97.23%**，完全匹配率 **79.11%**
- 6/E 缺类解决，B→5 混淆从 44 次降至 1 次
- 实验记录完整归档：`docs/experiment_log.md`
- 全部工具脚本 Git 安全清理（17 个脚本，路径脱敏）

### ⏳ 已知限制
- 部分器件刻印框数超过 4 个，当前 pipeline 假设 4 位编号（待后续处理）
- A/F-Z 类当前器件不涉及，暂不处理
- 通用 PaddleOCR CAR 仅 8.54%，已放弃通用方案

### 🔜 后续计划
- 处理超过 4 个框的器件（截断 / 动态长度 / 序列模型）
- A/F-Z 类扩展（如需支持更多器件再补标注）
- 引入公开工业字符数据集做跨域泛化验证

> CAR/CER 定义：CAR = 正确识别字符数/总字符数；CER = (替换+删除+插入错误数)/总字符数

> 说明：当前版本面向实验室验证，未覆盖真实产线环境下的长期稳定性与极端工况鲁棒性测试。

## 八、后续工作

- ~~接入 PaddleOCR 完成刻印字符端到端识别~~ ✅ 已完成（CAR 天花板 8.54%，已放弃通用方案）
- ~~优化预处理管道~~ ✅ 已完成（v2 版，CAR 8.54%）
- ~~训练单字符分类 CNN~~ ✅ 已完成（定制 CNN 95.23%）
- ~~ResNet-18 迁移学习验证~~ ✅ 已完成（v1 98.63% ⚠️，v2 99.19% ✅）
- ~~混合数据重训 YOLO + ResNet~~ ✅ 已完成（8.25）
- ~~新模型端到端评估~~ ✅ 已完成（CAR 97.23%，完全匹配率 79.11%）
- ~~脚本 Git 安全清理~~ ✅ 已完成（17 个脚本）
- **待做**：处理超过 4 个框的器件（评估影响范围 → 决定策略）
- **长期**：跨域泛化验证（不同材质/形状工业器件）

## 九、更新日志

### 2026-08-25（第十天）
- YOLOv8 新旧混合数据微调（mAP50 0.994 / mAP50-95 **0.921**，16ep early stopping）
- ResNet-18 v2 混合数据重训（8067 张，14 类，RGB 224×224，Val Acc **99.19%**）
- 新模型端到端评估 2212 张：CAR **97.23%**，完全匹配率 **79.11%**（旧模型 81.48% → 97.23%，+15.75%）
- 6/E 缺类解决（100% / 96.05%），B→5 混淆从 44 次降至 1 次
- 项目结构重组：`tools/training/`、`tools/tests/` 规范分类
- 全部脚本 Git 安全清理（17 个）：路径脱敏为示例相对路径 + 行末注释
- 记录已知限制：部分器件刻印框数超过 4 个（待后续处理）

### 2026-08-24（第九天）
- 完成端到端推理脚本 `main.py`（YOLOv8 检测 + ResNet-18 识别 → 输出 LabelImg 可读 YOLO txt，自动生成 36 类 classes.txt）
- 用 `main.py` 对 536 张新数据（同场景新器件）生成预测标签
- 解决 LabelImg 自动覆盖 `classes.txt` 的问题（采用临时目录隔离法）
- 完成 536 张新数据的手动修正标注
- 更新 README：补充 `main.py` 说明、当前进度、增量训练计划

### 2026-08-21（第八天）
- 完成 CNN 训练脚本（50ep / 100ep / +Random Erase 增强），最佳 Val Acc **95.23%**
- 完成 ResNet-18 迁移学习训练（100ep），最佳 Val Acc **98.63%**（⚠️ 输入 pipeline 不完善）
- 新增 `train_cnn_aug.py`（最终版 CNN）和 `train_resnet18.py`，输出至 `outputs/`
- 实验记录整理至 `docs/experiment_log.md`
- 更新 README：新增 CNN/ResNet 实验结果、项目结构、进度说明

### 2026-08-19（第七天）
- 完成 CNN 训练数据集制作脚本 `tools/tests/crop_cnn_dataset.py`
- 从 `labels_raw/` 原始逐字符标注 + `classes.txt` 裁剪 6231 张 48×48 灰度字符 ROI
- 实际覆盖 13 个类别：0(1693), 1(839), 2(139), 3(321), 4(763), 5(525), 7(255), 8(251), 9(177), B(120), C(526), D(615), E(7)
- 确认通用 PaddleOCR 路线已放弃，CNN 分类路线数据准备完毕，明日启动训练
- 更新 README 项目结构与进度说明

### 2026-08-14（第六天）
- 重构 `tools/` 目录结构：旧脚本归档至 `archive/`，核心脚本归入 `tests/`，输出统一至 `outputs/`
- 修正评估脚本 `evaluate_accuracy.py` 的过滤逻辑，修复 8/13 CAR 虚高问题（15.02% → 修正后基线 5.54%）
- 完成预处理 A/B 测试（v2/PlanA/PlanC 各 100 张），选定 v2 管道跑全量 1813 张：CAR = 8.54%（通用 PaddleOCR 真实天花板）
- 确认下一步方向：**放弃通用 PaddleOCR，自训 CNN 做 36 类字符分类**
- 所有脚本路径改为占位符，适配 Git 提交（不泄露本地路径）
- 新增 `learn/` 代码学习模块（不提交）
- 更新 README：全面修正实验数据（A/B 测试 + CAR 修正）、项目结构、后续计划与更新日志

### 2026-08-13（第五天）
- 新增 `tools/predict_batch_mp2.py`：2 进程并行批量识别脚本（Windows spawn 兼容）
- 新增 `tools/evaluate_accuracy.py`：标准 OCR 评估脚本（CAR/CER + 位置准确率 + 混淆矩阵）
- 批量测试 1813 张图，耗时 40.4 分钟，输出 `recognition_results_mp2.csv`
- **关键实验结论**：CAR=15.02%，CER=86.21%，4 位完全匹配率 0%
- **根因定位**：批量脚本相比单张脚本 `predict.py` 缺少轮廓实心化 + 尺寸归一化
- 单张脚本能对 1 个字符，批量脚本 CAR 仅 15%，确认瓶颈在预处理差异
- 下一步：加回轮廓实心化 + ROI 尺寸归一化到 48px + Padding 到 320px 后重测
- 更新 README 项目结构，补充 `predict_batch.py`、`predict_batch_mp2.py`、`evaluate_accuracy.py` 说明

### 2026-08-12（第四天）
- 为 `tools/labelsTOyolo.py` 补充注释
- 新建 `tools/crop_roi.py`，打通 YOLOv8 + PaddleOCR 端到端推理流程
- 确认当前 OCR 识别瓶颈：原始 ROI 对比度低，置信度仅 ~0.23
- 规划 ROI 图像增强方案（灰度 → 反色 → CLAHE → 二值化）
- 整理脚本学习笔记至 `notes.md`（不上传 GitHub）

> 说明：此前已完成模型训练与 OCR 环境验证，本次为复工后首次更新，重点为推理链路工程化与问题定位。

### 2026-08-07（第三天）
- 重构 `train.py`，移除过期数据清洗注释，新增 CPU 调试专用文档字符串（Docstring）
- 完善 README 中 CPU / GPU 训练说明，明确 CPU 指标仅用于流程验证
- 在 `requirements.txt` 中补充 PyTorch CPU / GPU 安装指引
- 确立“代码管逻辑、文档管使用、notes 管实验”的信息分层规范
- 将 README 命令与 YAML 配置统一包裹为代码块，提升可读性
- 明确 `data.yaml` 由使用者自行创建，`weekly_reports/` 不纳入版本管理

> 说明：今日重点由功能验证转向工程规范化，涵盖脚本精简、依赖说明、文档排版与使用边界澄清。

### 2026-08-06（第二天）
- 明确项目为个人验证性质，暂不追求工业级鲁棒性
- 确认标注策略：手写体编号刻意不标，视为背景，避免污染 OCR 输入
- 梳理 YOLO 与 OCR 分工：YOLO 负责定位，OCR 负责识别，后处理负责过滤误检
- 制定迭代路线：优先完成 OCR 后处理方案（一期），视时间再优化检测阶段手写体排斥能力（二期）
- 补充 LabelImg 字母映射导致类别 ID 非 0 的隐蔽问题说明
- 新增 `tools/test_single_image.py`，用于 PaddleOCR 3.x 单图推理验证
- 确认 PaddleOCR 初始化需添加 `enable_mkldnn=False`，以规避 PaddleX/oneDNN/PIR 相关 `NotImplementedError`
- 初步验证：PP-OCRv6 对原始金属刻印图直接推理可能返回 0 个文本区域
- 明确后续方向：不能依赖通用 OCR 直接识别原始工业刻印图，需要 YOLO ROI 裁剪 + 图像增强/二值化/CLAHE 后再识别
- 新增 `requirements.txt`，锁定 PaddlePaddle==3.3.1 与 PaddleOCR==3.7.0，确保环境可复现

### 2026-08-05（第一天）
- 完成 YOLOv8 模型训练、验证与文档初稿
- 完成数据集划分与标注预处理脚本开发（`tools/labelsTOyolo.py`）

## ⚠️ 关于 CPU / GPU 的说明

- 本项目默认推荐使用 GPU 进行训练与评估；
- `train.py` 中 `device="cpu"` **仅为本地调试用途**，不代表模型设计或性能基准；
- CPU 环境下的指标仅用于验证训练流程与数据正确性；
- 实际性能、推理速度与 mAP 均以 GPU（RTX 3080 Ti）结果为准。