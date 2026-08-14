# YOLOv8 类子弹头状器件刻印区域检测

## 一、项目背景

在工业质检场景中，需要在金属器件尖端表面识别凹印/凸印编号。该类目标存在以下难点：

- 金属反光强，局部过曝
- 刻印字符尺寸小、对比度低
- 部分样本无刻印（负样本）
- 图像存在倾斜、旋转与尺度变化

> 注：本项目为个人验证性质，基于 1813 张样本完成算法原型验证，不涉及真实产线部署。

## 二、数据集说明

- 来源：实习单位内部采集（受保密协议限制，不公开原始数据）
- 规模：1813 张工业图像
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

pip install -r requirements.txt

> 建议使用 GPU 环境以获得正常训练速度与性能。CPU 环境仅适合流程验证。

主要依赖：

- Python>=3.8
- PyTorch>=2.0
- Ultralytics YOLOv8
- OpenCV
- paddlepaddle==3.3.1
- paddleocr==3.7.0
- Matplotlib

## 四、快速开始

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

### 5.1 本地 CPU 验证结果（流程正确性验证）

- 目的：验证训练流水线连通性及标签解析一致性（排除静默失效风险）
- 硬件：AMD Ryzen 7 7730U（无独显）
- mAP50：0.994
- mAP50-95：0.889
- 推理速度：127 ms / 张

> 注：CPU 结果仅用于功能验证，不代表模型性能上限。

### 5.2 GPU 训练结果（AutoDL RTX 3080 Ti，50 Epochs）

- Epochs：50
- mAP50：0.994
- mAP50-95：0.910
- Precision：0.986
- Recall：0.979
- 推理速度：0.6 ms / 张
- 训练耗时：~12 min

> 注：GPU 结果为实际性能参考，后续实验与评估均基于此配置。

**训练曲线**

![YOLOv8 训练曲线](./assets/results.png)

### 5.3 预处理 A/B 测试与选型

为突破通用 PaddleOCR 在工业刻印 ROI 上的识别瓶颈，对三个预处理方案进行了 100 张图像的 A/B 测试：

- **v2（CLAHE + 9×9 模糊 + 3×3 实心化 + 48px 归一化 + 320 Padding）**：CAR = 13.35%（最优）
- **Plan A（基线 + 2×2 膨胀，无归一化）**：CAR = 9.94%
- **Plan C（5×5 模糊 + 开运算 + 64px）**：CAR = 5.68%

基于 A/B 测试结果，选定 **v2 管道** 进行 1813 张全量测试，最终 CAR = **8.54%**。确认此为通用 PaddleOCR 在当前数据集上的真实天花板。

## 六、项目结构
```text
├── assets/                    # 静态资源（README 用图）
│   └── results.png            # 训练曲线图
├── learn/                     # 代码学习脚本（不提交）
├── weekly_reports/            # 个人实习周报（不提交）
├── notes.md                   # 个人学习笔记（不提交）
├── requirements.txt           # 项目依赖清单
├── data.yaml.example          # 数据配置模板（参考用）
├── data.yaml                  # 本地私有配置（不提交，由 data.yaml.example 复制生成）
├── train.py                   # 训练脚本
├── tools/                     # 工具脚本目录
│   ├── archive/               # 历史调试脚本归档（不提交）
│   ├── outputs/               # 批量识别 CSV 输出（不提交）
│   ├── tests/                 # 当前核心工具链
│   │   ├── test_env.py        # 环境测试脚本
│   │   └── test_single_image.py # PaddleOCR 单图推理测试脚本
│   ├── labelsTOyolo.py        # 标签标准化脚本（统一类别 ID 为 0）
│   ├── crop_roi.py            # YOLO 检测 + PaddleOCR 识别端到端推理脚本
│   ├── predict_batch.py       # 批量识别（v2 预处理管道，2 进程并行）
│   └── evaluate_accuracy.py   # OCR 评估脚本（计算 CAR/CER/混淆矩阵）
├── runs/                      # 训练输出目录（不提交，含 best.pt / results.png 等）
├── yolov8n.pt                 # YOLOv8n 官方预训练权重（不提交，首次 train 时自动下载）
└── README.md                  # 项目说明文档
```
说明：
- 带（不提交）标记的文件/目录已加入 `.gitignore`，不会上传至 GitHub；
- 实际训练使用 `data.yaml`，该文件仅存在于本地，由使用者根据环境自行创建，不会被版本管理；
- `weekly_reports/` 存放个人实习周报，不纳入版本管理与项目交付范围；
- 使用者需根据 `data.yaml.example` 创建本地配置。

## 七、当前进度说明

### ✅ 已完成
- 刻印区域检测模型训练完成（mAP50: 0.994, mAP50-95: 0.910）
- PaddleOCR 3.x 环境与新 API 跑通
- YOLO 检测 → ROI 裁剪 → OCR 识别端到端链路打通
- 批量识别脚本 `tools/tests/predict_batch.py` 开发完成（v2 预处理管道，2 进程并行）
- 评估脚本 `tools/evaluate_accuracy.py` 开发完成（CAR/CER 双指标）

### ❌ 当前瓶颈（2026-08-14 修正）

> ⚠️ 8/13 报告的 CAR=15.02% 为虚高（评估脚本未过滤导致），已修正。

- 修正后基线（带过滤）：CAR = 5.54%
- v2 预处理管道全量（1813 张）：**CAR = 8.54%**（真实天花板）
- 4 位完全匹配率：0%
- 主要错误模式：OCR 输出 `?`（DB 检测模型在 ROI 上直接返回空），而非识别错字符
- 根因：通用 PaddleOCR 对工业小字符、强反光 ROI 的 DB 检测能力不足，
  预处理优化已达瓶颈（CAR 最高 ~8.5%）

### ⏳ 下一步计划（已确定）
- **放弃通用 PaddleOCR 识别，改为自训 CNN 字符分类器（36 类：0-9 + A-Z）**
- 理由：PaddleOCR 的 DB 检测模块在你的 ROI 上直接返回空，预处理无法绕过此限制
- 预计：用 YOLO 裁剪的 ROI 作为训练数据，训练轻量 CNN 做单字符分类

> CAR/CER 定义：CAR = 正确识别字符数/总字符数；CER = (替换+删除+插入错误数)/总字符数

> 说明：当前版本面向实验室验证，未覆盖真实产线环境下的长期稳定性与极端工况鲁棒性测试。

## 八、后续工作

- ~~接入 PaddleOCR 完成刻印字符端到端识别~~ ✅ 已完成（CAR 真实天花板 ~8.5%，达瓶颈）
- ~~优化预处理管道（轮廓实心化 + 尺寸归一化 + Padding）~~ ✅ 已完成（v2 版，CAR 8.54%）
- **短期（进行中）**：训练单字符分类 CNN（36 类：0-9 + A-Z），替代 PaddleOCR 识别（详见瓶颈分析）
- **中期**：若 CNN 分类达标，重新跑全量 1813 张，更新 CAR/CER 评估
- **长期**：针对倾斜刻印设计自适应增强策略；评估是否需将手写体作为强负样本参与检测训练

## 九、更新日志

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