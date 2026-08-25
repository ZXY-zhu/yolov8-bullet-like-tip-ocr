"""
tools/training/train_resnet18_v2.py
用途：ResNet-18 迁移学习字符分类器（v2，AutoDL 环境）
说明：
- 数据集：dataset_cnn_rgb/（36 类工业刻印字符，RGB 三通道）
- 输入：RGB → Resize 224×224
- 预训练：ImageNet1K V1
- 增强：翻转 + 旋转 + 仿射变换 + ImageNet 归一化
- 调度：Adam(lr=3e-4) + CosineAnnealingLR
- 运行方式：直接改下面 2 个路径，然后 python tools/training/train_resnet18_v2.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ========== 改这里 ==========
DATASET_ROOT = "dataset_cnn_rgb"                            # 改为实际数据集目录路径
OUTPUT_DIR = "outputs"                                      # 改为实际模型输出目录路径
# ==========================

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 3e-4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_RATIO = 0.2
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"🚀 设备: {DEVICE}")
print(f"📁 数据集: {DATASET_ROOT}")

class CharDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 只保留非空文件夹，自动适配实际有数据的类
        all_classes = sorted(os.listdir(root_dir))
        self.classes = []
        self.class_to_idx = {}
        for i, name in enumerate(all_classes):
            class_dir = os.path.join(root_dir, name)
            if not os.path.isdir(class_dir):
                continue
            if not any(f.endswith((".png", ".jpg", ".jpeg")) for f in os.listdir(class_dir)):
                continue
            self.classes.append(name)
            self.class_to_idx[name] = len(self.classes) - 1

        print(f"📋 实际使用的类 ({len(self.classes)}): {self.classes}")

        all_paths, all_labels = [], []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.endswith((".png", ".jpg", ".jpeg")):
                    all_paths.append(os.path.join(class_dir, img_name))
                    all_labels.append(class_idx)

        indices = list(range(len(all_paths)))
        train_idx, val_idx = train_test_split(
            indices, test_size=VAL_RATIO, stratify=all_labels, random_state=RANDOM_SEED
        )
        selected = train_idx if split == "train" else val_idx
        self.image_paths = [all_paths[i] for i in selected]
        self.labels = [all_labels[i] for i in selected]

        print(f"📊 [{split}] {len(self.image_paths)} 张")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = CharDataset(DATASET_ROOT, "train", train_transform)
val_ds = CharDataset(DATASET_ROOT, "val", val_transform)
num_classes = len(train_ds.classes)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

class CharResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)

model = CharResNet(num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0.0
model_path = os.path.join(OUTPUT_DIR, "resnet18_best.pth")

print(f"\n🔥 训练 ResNet-18 ({num_classes} 类)...")

for epoch in range(EPOCHS):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(train_loader, desc=f"E{epoch+1}/{EPOCHS}[Train]"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    train_acc = 100. * correct / total

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"E{epoch+1}/{EPOCHS}[Val]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, pred = outputs.max(1)
            val_total += labels.size(0)
            val_correct += pred.eq(labels).sum().item()
    val_acc = 100. * val_correct / val_total

    scheduler.step()
    print(f"Epoch [{epoch+1}/{EPOCHS}] Train: {train_acc:.2f}% Val: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print(f"  ✅ 新最佳! Val Acc: {val_acc:.2f}%")

print(f"\n🎉 完成！最佳 Val Acc: {best_val_acc:.2f}%")
print(f"📁 权重: {model_path}")