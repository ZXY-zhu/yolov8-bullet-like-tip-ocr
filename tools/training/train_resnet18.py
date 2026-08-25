"""
tools/training/train_resnet18.py
用途：ResNet-18 迁移学习字符分类器
说明：
- 数据集：dataset_cnn/（13 类工业刻印字符，6189 张）
- 输入：灰度 48×48 → 转 RGB → Resize 224×224（⚠️ 存在上采样信息损失）
- 预训练：ImageNet1K V1
- 增强：翻转 + 旋转 + 颜色抖动 + ImageNet 归一化
- 调度：Adam(lr=1e-4) + CosineAnnealingLR
- 最佳 Val Acc：98.63%（Epoch 70, 100ep）
- 运行方式：直接改下面 DATASET_ROOT，然后 python tools/training/train_resnet18.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ========== 1. 路径与超参数 ==========
DATASET_ROOT = "dataset_cnn"                               # 改为实际数据集目录路径
OUTPUT_DIR = "outputs"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_RATIO = 0.2
RANDOM_SEED = 42
print(f"🚀 使用设备: {DEVICE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 2. Dataset ==========
class CharDataset(Dataset):
    """灰度 PNG → 转 RGB 三通道 → PIL Image（适配 ResNet 预训练权重）"""
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        all_paths = []
        all_labels = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.endswith(".png"):
                    all_paths.append(os.path.join(class_dir, img_name))
                    all_labels.append(class_idx)

        indices = list(range(len(all_paths)))
        train_idx, val_idx = train_test_split(
            indices, test_size=VAL_RATIO, stratify=all_labels, random_state=RANDOM_SEED
        )

        if split == "train":
            selected = train_idx
        else:
            selected = val_idx

        for i in selected:
            self.image_paths.append(all_paths[i])
            self.labels.append(all_labels[i])

        print(f"📊 [{split}] 共 {len(self.image_paths)} 张图, {len(self.classes)} 个类别")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# ========== 3. 模型 ==========
class CharResNet(nn.Module):
    """ResNet-18 + ImageNet 预训练 + 替换分类头"""
    def __init__(self, num_classes):
        super(CharResNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# ========== 4. 数据增强 ==========
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ========== 5. 训练循环 ==========
def train_epoch(model, dataloader, criterion, optimizer, device):
    """RGB 3 通道输入，ToTensor + Normalize 已处理归一化"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})

    return running_loss / len(dataloader), 100 * correct / total

# ========== 6. 验证循环 ==========
def val_epoch(model, dataloader, criterion, device):
    """单轮验证"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'val_acc': f'{100*correct/total:.2f}%'})

    return running_loss / len(dataloader), 100 * correct / total

# ========== 7. 主函数 ==========
if __name__ == "__main__":
    train_dataset = CharDataset(root_dir=DATASET_ROOT, split="train", transform=train_transform)
    val_dataset = CharDataset(root_dir=DATASET_ROOT, split="val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = CharResNet(num_classes=len(train_dataset.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    model_path = os.path.join(OUTPUT_DIR, "resnet18_char_classifier_best.pth")

    print("\n🔥 开始训练（ResNet-18 预训练微调）...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"  ✅ 新的最佳模型！Val Acc: {val_acc:.2f}%")

    print(f"\n🎉 训练完成！最佳验证集准确率: {best_val_acc:.2f}%")
    print(f"📁 最佳模型已保存至: {model_path}")