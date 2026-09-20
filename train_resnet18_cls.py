import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from collections import Counter

# ===================== 路径配置 =====================
TRAIN_DIR = "/home/ytkj/work/hangyq/Projects/UNet/zaosheng/data_clean/train"
VAL_DIR = "/home/ytkj/work/hangyq/Projects/UNet/zaosheng/data_clean/val"
PRETRAINED_PATH = "/home/ytkj/work/hangyq/Projects/UNet/pretrained/resnet18-f37072fd.pth"

SAVE_DIR = os.path.expanduser("~/zaosheng/train_result")
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 训练参数 =====================
NUM_CLASSES = 26
BATCH_SIZE = 8
EPOCHS = 25
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# ===================== 数据增强 =====================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===================== 数据集 =====================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))
print("Classes:", train_dataset.classes)

# ===================== 解决类别不平衡（修复版）=====================
targets = train_dataset.targets
class_counts = Counter(targets)

print("Class distribution:", class_counts)

weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [weights[label] for label in targets]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# ===================== DataLoader =====================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,       # ⚠️ 先设0避免卡死
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ===================== 模型 =====================
model = models.resnet18(weights=None)

print("Loading pretrained weights...")
state_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)

model = model.to(DEVICE)

# ===================== 损失 & 优化 =====================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# ===================== 记录 =====================
train_loss_list, train_acc_list = [], []
val_loss_list, val_acc_list = [], []

# ===================== 训练 =====================
best_acc = 0.0

print("\n🚀 Start Training...\n")

for epoch in range(EPOCHS):
    model.train()
    train_loss, corr, total = 0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = out.max(1)
        corr += (pred == labels).sum().item()
        total += labels.size(0)

    train_acc = corr / total
    avg_train_loss = train_loss / len(train_loader)

    # ===== 验证 =====
    model.eval()
    val_loss, corr, total = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)

            val_loss += criterion(out, labels).item()
            _, pred = out.max(1)
            corr += (pred == labels).sum().item()
            total += labels.size(0)

    val_acc = corr / total
    avg_val_loss = val_loss / len(val_loader)

    train_loss_list.append(avg_train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1:2d} | train_acc={train_acc:.2%} | val_acc={val_acc:.2%}")

    scheduler.step(avg_val_loss)

    # ===== 保存模型 =====
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_noise_cls.pth"))

# ===================== 保存曲线 =====================
plt.figure()
plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list, label="Val Loss")
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"))
plt.close()

plt.figure()
plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list, label="Val Acc")
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "acc_curve.png"))
plt.close()

# ===================== 分类报告 =====================
print("\n📊 Generating classification report...")

model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_noise_cls.pth")))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        out = model(imgs)
        preds = torch.argmax(out, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

report = classification_report(
    all_labels,
    all_preds,
    target_names=val_dataset.classes,
    digits=4
)

with open(os.path.join(SAVE_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)

print("\n🎉 训练完成！")
print(f"📂 保存路径: {SAVE_DIR}")
print(f"🏆 最佳验证准确率: {best_acc:.2%}")