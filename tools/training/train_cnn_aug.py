"""
tools/training/train_cnn_aug.py
用途：定制 CNN 字符分类器（带 Random Erase 增强）
说明：
- 数据集：dataset_cnn/（13 类工业刻印字符，6189 张灰度 48×48）
- 模型：2 层 Conv + Dropout，从零训练
- 增强：Random Erase（p=0.5，模拟脏污遮挡）
- 最佳 Val Acc：95.23%（100 epoch）
- 运行方式：直接改下面 DATASET_ROOT，然后 python tools/training/train_cnn_aug.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ========== 1. 路径与超参数配置 ==========
DATASET_ROOT = "dataset_cnn"                               # 改为实际数据集目录路径
OUTPUT_DIR = "outputs"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
IMG_SIZE = 48
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_RATIO = 0.2
RANDOM_SEED = 42
print(f"🚀 使用设备: {DEVICE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ========== 2. 自定义 Dataset ==========
class CharDataset(Dataset):
    """工业刻印字符数据集。读取 dataset_cnn/{字符}/ 下的 PNG，按 8:2 分层拆分。"""
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
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            image = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


# ========== 3. CNN 模型 ==========
class CharCNN(nn.Module):
    """轻量字符分类 CNN：2×Conv+ReLU+MaxPool → Flatten → 128 → Dropout(0.5) → num_classes"""
    def __init__(self, num_classes):
        super(CharCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ========== 4. Random Erase 增强 ==========
class RandomErase:
    """随机遮挡：以 p 概率将图片上一块矩形区域置为 0（黑），模拟工业脏污遮挡。"""
    def __init__(self, p=0.5, area_ratio=(0.02, 0.15)):
        self.p = p
        self.area_ratio = area_ratio

    def __call__(self, image):
        if np.random.rand() > self.p:
            return image

        h, w = image.shape
        area = h * w
        erase_area = np.random.uniform(*self.area_ratio) * area
        erase_h = int(np.sqrt(erase_area * w / h))
        erase_w = int(erase_area / erase_h)
        erase_h = min(erase_h, h)
        erase_w = min(erase_w, w)

        y = np.random.randint(0, h - erase_h + 1)
        x = np.random.randint(0, w - erase_w + 1)

        image[y:y+erase_h, x:x+erase_w] = 0
        return image


# ========== 5. 训练循环 ==========
def train_epoch(model, dataloader, criterion, optimizer, device):
    """单轮训练：灰度单通道 → unsqueeze(1) → 归一化到 [0,1]"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        images = images.unsqueeze(1).float() / 255.0

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})

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
            images = images.unsqueeze(1).float() / 255.0
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'val_acc': f'{100 * correct / total:.2f}%'})

    return running_loss / len(dataloader), 100 * correct / total


# ========== 7. 主函数 ==========
if __name__ == "__main__":
    train_transform = transforms.Compose([
        RandomErase(p=0.5, area_ratio=(0.02, 0.15)),
    ])
    val_transform = transforms.Compose([])

    train_dataset = CharDataset(root_dir=DATASET_ROOT, split="train", transform=train_transform)
    val_dataset = CharDataset(root_dir=DATASET_ROOT, split="val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CharCNN(num_classes=len(train_dataset.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    model_path = os.path.join(OUTPUT_DIR, "cnn_char_classifier_best_aug.pth")

    print("\n🔥 开始训练（带 Random Erase 增强）...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, DEVICE)

        print(f"Epoch [{epoch+1}/{EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"  ✅ 新的最佳模型！Val Acc: {val_acc:.2f}%")

    print(f"\n🎉 训练完成！最佳验证集准确率: {best_val_acc:.2f}%")
    print(f"📁 最佳模型已保存至: {model_path}")