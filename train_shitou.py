import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# ==================== 配置 ====================
class Config:
    TRAIN_IMG_DIR = '/home/ytkj/work/hangyq/Projects/ultralytics-main/shitou/images/train'
    TRAIN_MASK_DIR = '/home/ytkj/work/hangyq/Projects/ultralytics-main/shitou/masks/train'
    TEST_IMG_DIR = '/home/ytkj/work/hangyq/Projects/ultralytics-main/shitou/images/test'
    TEST_MASK_DIR = '/home/ytkj/work/hangyq/Projects/ultralytics-main/shitou/masks/test'

    EPOCHS = 200
    BATCH_SIZE = 10
    LEARNING_RATE = 1e-4
    IMG_SIZE = (384, 672)
    NUM_CLASSES = 4  # 0:背景, 1:coal, 2:gangue, 3:other
    NUM_WORKER = 4
    DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    SAVE_DIR = '/home/ytkj/work/hangyq/Projects/ultralytics-main/shitou/unet'

    SEED = 42
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

def resize_image_and_mask(image, mask, img_size):
    h, w = img_size
    image = TF.resize(image, (h, w))
    mask = TF.resize(mask, (h, w), interpolation=Image.NEAREST)
    return image, mask

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(Config.SEED)

# ==================== UNet（多分类） ====================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, num_classes, 1)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        return self.out_conv(d1)

# ==================== 数据集（jpg→png） ====================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.images = sorted(os.listdir(image_dir))
        self.normalize = transforms.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        base = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, base + ".png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image, mask = resize_image_and_mask(image, mask, self.img_size)
        image = TF.to_tensor(image)
        image = self.normalize(image)

        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask

# ==================== 指标：mIoU + Precision + Recall ====================
def calculate_miou_precision_recall(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1)
    ious = []
    precisions = []
    recalls = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter
        iou = (inter + 1e-6) / (union + 1e-6)
        ious.append(iou.item())
        precision = inter / (p.sum() + 1e-6)
        precisions.append(precision.item())
        recall = inter / (t.sum() + 1e-6)
        recalls.append(recall.item())
    return np.mean(ious), np.mean(precisions), np.mean(recalls)

# ==================== 训练/验证 ====================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_miou, total_prec, total_recall = 0.0, 0.0, 0.0, 0.0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        if outputs.shape[2:] != masks.shape[1:]:
            outputs = nn.functional.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        miou, prec, recall = calculate_miou_precision_recall(outputs, masks, Config.NUM_CLASSES)
        total_loss += loss.item()
        total_miou += miou
        total_prec += prec
        total_recall += recall
    return total_loss/len(dataloader), total_miou/len(dataloader), total_prec/len(dataloader), total_recall/len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_miou, total_prec, total_recall = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            if outputs.shape[2:] != masks.shape[1:]:
                outputs = nn.functional.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, masks)
            miou, prec, recall = calculate_miou_precision_recall(outputs, masks, Config.NUM_CLASSES)
            total_loss += loss.item()
            total_miou += miou
            total_prec += prec
            total_recall += recall
    return total_loss/len(dataloader), total_miou/len(dataloader), total_prec/len(dataloader), total_recall/len(dataloader)

# ==================== 可视化（彩色4类） ====================
def visualize_results(model, dataloader, device, save_dir, epoch, num_samples=3):
    model.eval()
    colors = np.array([
        [0, 0, 0],       # 0:背景 黑
        [255, 0, 0],     # 1:coal 红
        [0, 255, 0],     # 2:gangue 绿
        [0, 0, 255]      # 3:other 蓝
    ], dtype=np.uint8)
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(Config.NORM_MEAN, Config.NORM_STD)],
        std=[1/s for s in Config.NORM_STD]
    )
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_samples:
                break
            images = images.to(device)
            outputs = model(images)
            if outputs.shape[2:] != masks.shape[1:]:
                outputs = nn.functional.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
            pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            mask_gt = masks[0].numpy()
            img = inv_normalize(images[0]).cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            gt_color = colors[mask_gt]
            pred_color = colors[pred]
            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Image")
            axes[i, 0].axis('off')
            axes[i, 1].imshow(gt_color)
            axes[i, 1].set_title("GT Mask")
            axes[i, 1].axis('off')
            axes[i, 2].imshow(pred_color)
            axes[i, 2].set_title("Pred Mask")
            axes[i, 2].axis('off')
    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(save_dir)/f'epoch_{epoch}.png', dpi=150)
    plt.close()

# ==================== 绘制训练曲线：Loss / mIoU / Precision / Recall ====================
def plot_training_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(16, 10))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_miou'], label='Train mIoU')
    plt.plot(epochs, history['val_miou'], label='Val mIoU')
    plt.title('mIoU')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_prec'], label='Train Precision')
    plt.plot(epochs, history['val_prec'], label='Val Precision')
    plt.title('Precision')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['train_recall'], label='Train Recall')
    plt.plot(epochs, history['val_recall'], label='Val Recall')
    plt.title('Recall')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

# ==================== 主函数 ====================
def main():
    Path(Config.SAVE_DIR).mkdir(parents=True, exist_ok=True)
    print("="*70)
    print("UNet 4-class (背景/coal/gangue/other)")
    print("="*70)
    train_dataset = SegmentationDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR, Config.IMG_SIZE)
    val_dataset = SegmentationDataset(Config.TEST_IMG_DIR, Config.TEST_MASK_DIR, Config.IMG_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKER, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKER, pin_memory=True)
    model = UNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    history = {
        'train_loss': [], 'train_miou': [], 'train_prec': [], 'train_recall': [],
        'val_loss': [], 'val_miou': [], 'val_prec': [], 'val_recall': []
    }
    best_miou = 0.0
    for epoch in range(1, Config.EPOCHS+1):
        tloss, tmiou, tprec, trecall = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        vloss, vmiou, vprec, vrecall = validate(model, val_loader, criterion, Config.DEVICE)
        scheduler.step(vloss)
        history['train_loss'].append(tloss)
        history['train_miou'].append(tmiou)
        history['train_prec'].append(tprec)
        history['train_recall'].append(trecall)
        history['val_loss'].append(vloss)
        history['val_miou'].append(vmiou)
        history['val_prec'].append(vprec)
        history['val_recall'].append(vrecall)

        print(f"Epoch {epoch:3d} | "
              f"Train Loss {tloss:.3f} | mIoU {tmiou:.3f} | Prec {tprec:.3f} | Recall {trecall:.3f} | "
              f"Val Loss {vloss:.3f} | mIoU {vmiou:.3f} | Prec {vprec:.3f} | Recall {vrecall:.3f}")

        if vmiou > best_miou:
            best_miou = vmiou
            torch.save(model.state_dict(), Path(Config.SAVE_DIR)/"best.pth")

        if epoch % 20 == 0 or epoch == 1:
            visualize_results(model, val_loader, Config.DEVICE, Config.SAVE_DIR, epoch)

    print(f"\nBest Val mIoU: {best_miou:.4f}")
    plot_training_curves(history, os.path.join(Config.SAVE_DIR, 'training_curves.png'))

if __name__ == "__main__":
    main()