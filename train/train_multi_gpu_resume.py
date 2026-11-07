import os
import json
import random
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

from model.unet import *
from model.unet_eca import *


# ==================== 配置部分 ====================
class Config:
    # 数据集路径
    TRAIN_IMG_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/xuzhou/train_images'
    TRAIN_MASK_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/xuzhou/train_masks'
    TEST_IMG_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/xuzhou/test_images'
    TEST_MASK_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/xuzhou/test_masks'

    # 训练参数
    EPOCHS = 120
    BATCH_SIZE = 12
    LEARNING_RATE = 1e-4
    IMG_SIZE = (384, 672)
    NUM_CLASSES = 2
    NUM_WORKER = 6

    # 主设备
    DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    GPU_IDS = [2, 3]  # 例如 [0,1] / [2,3]

    SAVE_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/result/unet_1_public_0525'
    SEED = 64
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    SHIFT_RANGE = (-50, 50)

    RESUME = True  # True=续训，False=从头开始

    # 建议使用 latest_checkpoint.pth 作为“最新断点”
    CHECKPOINT_PATH = "/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/result/unet_1_public_0525/checkpoint_epoch_70.pth"

    # 如果你想从旧 checkpoint_epoch_70.pth 恢复一次，可以临时改成下面这个路径：
    # CHECKPOINT_PATH = "/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/result/unet_1_public_0525/checkpoint_epoch_70.pth"


# ==================== 工具函数 ====================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


def resize_image_and_mask(image, mask, img_size):
    """安全调整图像和掩码尺寸，保持同步"""
    h, w = img_size
    image = TF.resize(image, (h, w))
    mask = TF.resize(mask, (h, w), interpolation=Image.NEAREST)
    return image, mask


def random_shift(image, mask, shift_range):
    """
    对PIL格式的图片和掩码执行随机上下/左右平移
    """
    shift_x = random.randint(shift_range[0], shift_range[1])
    shift_y = random.randint(shift_range[0], shift_range[1])

    image = TF.affine(
        image, angle=0, translate=(shift_x, shift_y),
        scale=1.0, shear=0, fill=0
    )
    mask = TF.affine(
        mask, angle=0, translate=(shift_x, shift_y),
        scale=1.0, shear=0, fill=0
    )
    return image, mask


def get_model_state_dict(model):
    """兼容单卡/多卡，获取真实模型参数"""
    return model.module.state_dict() if hasattr(model, 'module') else model.state_dict()


def save_checkpoint(path, model, optimizer, scheduler, epoch, history, best_val_iou, best_epoch):
    """保存完整断点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': get_model_state_dict(model),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'history': history,
        'best_val_iou': best_val_iou,
        'best_epoch': best_epoch
    }
    torch.save(checkpoint, path)


def load_model_state_compat(model, state_dict):
    """
    兼容单卡/多卡 checkpoint 的权重加载
    """
    model_state_dict = model.state_dict()

    ckpt_keys = list(state_dict.keys())
    model_keys = list(model_state_dict.keys())

    if len(ckpt_keys) == 0:
        raise ValueError("checkpoint 中 model_state_dict 为空，无法加载")

    ckpt_has_module = ckpt_keys[0].startswith('module.')
    model_has_module = model_keys[0].startswith('module.')

    new_state_dict = OrderedDict()

    if ckpt_has_module == model_has_module:
        new_state_dict = state_dict
    elif ckpt_has_module and not model_has_module:
        # checkpoint 是多卡保存，当前是单卡
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '', 1)] = v
    elif not ckpt_has_module and model_has_module:
        # checkpoint 是单卡保存，当前是多卡
        for k, v in state_dict.items():
            new_state_dict['module.' + k] = v

    model.load_state_dict(new_state_dict)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    加载断点，兼容旧版 checkpoint 缺字段的情况
    """
    print(f"🔄 从断点恢复：{path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if 'model_state_dict' not in checkpoint:
        raise KeyError("checkpoint 中缺少 'model_state_dict'，无法恢复模型")

    load_model_state_compat(model, checkpoint['model_state_dict'])

    if optimizer is not None:
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("⚠️ checkpoint 中未找到 optimizer_state_dict，优化器将重新初始化")

    loaded_epoch = checkpoint.get('epoch', 0)

    if scheduler is not None:
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("⚠️ checkpoint 中未找到 scheduler_state_dict，学习率调度器将重新初始化")
            # 如果希望尽量按 epoch 对齐，可取消注释下一行
            # scheduler.last_epoch = loaded_epoch

    history = checkpoint.get('history', {
        'train_loss': [], 'train_iou': [], 'train_dice': [],
        'val_loss': [], 'val_iou': [], 'val_dice': []
    })
    best_val_iou = checkpoint.get('best_val_iou', 0.0)
    best_epoch = checkpoint.get('best_epoch', 0)
    start_epoch = loaded_epoch + 1

    print(f"✅ 恢复成功，从 Epoch {start_epoch} 继续训练")
    return start_epoch, history, best_val_iou, best_epoch


# ==================== 数据集类 ====================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size, is_train=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.images = sorted(os.listdir(image_dir))
        self.is_train = is_train

        self.normalize = transforms.Normalize(
            mean=Config.NORM_MEAN,
            std=Config.NORM_STD
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 1. resize
        image, mask = resize_image_and_mask(image, mask, self.img_size)

        # 2. 数据增强（仅训练集）
        if self.is_train:
            image, mask = random_shift(image, mask, Config.SHIFT_RANGE)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        # 3. tensor + normalize
        image = TF.to_tensor(image)
        image = self.normalize(image)

        mask = torch.from_numpy(np.array(mask)).long()
        mask = (mask > 0).long()

        return image, mask


# ==================== 损失函数和评估指标 ====================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1).float()

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.25, dice_weight=0.75):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        target = target.float()
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(pred, target)


def calculate_iou(pred, target, threshold=0.7):
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def calculate_dice(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    return dice.item()


# ==================== 训练和验证函数 ====================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0

    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)

        if outputs.shape[2:] != masks.shape[1:]:
            outputs = nn.functional.interpolate(
                outputs,
                size=(masks.shape[1], masks.shape[2]),
                mode='bilinear',
                align_corners=False
            )

        loss = criterion(outputs, masks.unsqueeze(1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += calculate_iou(outputs, masks.unsqueeze(1))
        total_dice += calculate_dice(outputs, masks.unsqueeze(1))

    n_batches = len(dataloader)
    return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)

            if outputs.shape[2:] != masks.shape[1:]:
                outputs = nn.functional.interpolate(
                    outputs,
                    size=(masks.shape[1], masks.shape[2]),
                    mode='bilinear',
                    align_corners=False
                )

            loss = criterion(outputs, masks.unsqueeze(1))

            total_loss += loss.item()
            total_iou += calculate_iou(outputs, masks.unsqueeze(1))
            total_dice += calculate_dice(outputs, masks.unsqueeze(1))

    n_batches = len(dataloader)
    return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches


# ==================== 可视化函数 ====================
def visualize_results(model, dataloader, device, save_dir, epoch, num_samples=5):
    model.eval()

    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(Config.NORM_MEAN, Config.NORM_STD)],
        std=[1 / s for s in Config.NORM_STD]
    )

    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_samples:
                break

            images = images.to(device)
            outputs = model(images)

            if outputs.shape[2:] != masks.shape[1:]:
                outputs = nn.functional.interpolate(
                    outputs,
                    size=masks.shape[1:],
                    mode='bilinear',
                    align_corners=False
                )

            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()

            img = inv_normalize(images[0].cpu()).numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)

            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'原图\nSample {i + 1}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(masks[0].numpy(), cmap='gray')
            axes[i, 1].set_title('真实 Mask')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(preds[0, 0], cmap='gray')
            axes[i, 2].set_title('预测 Mask')
            axes[i, 2].axis('off')

            overlay = img.copy()
            overlay[preds[0, 0] > 0.5] = [0, 1, 0]
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('叠加效果')
            axes[i, 3].axis('off')

    plt.tight_layout()
    save_path = Path(save_dir) / f'epoch_{epoch}_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  可视化结果已保存：{save_path}")


def plot_training_history(history, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history['train_iou'], 'b-', label='Train IoU')
    axes[1].plot(epochs, history['val_iou'], 'r-', label='Val IoU')
    axes[1].set_title('IoU')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, history['train_dice'], 'b-', label='Train Dice')
    axes[2].plot(epochs, history['val_dice'], 'r-', label='Val Dice')
    axes[2].set_title('Dice Coefficient')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Dice')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    save_path = Path(save_dir) / 'training_history.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练历史已保存：{save_path}")


# ==================== 主训练函数 ====================
def main():
    print("=" * 70)
    print("UNet 分割训练脚本（BCE:Dice=1:3 + 归一化 + Shift+Flip增强 + 断点续训）")
    print("=" * 70)

    Path(Config.SAVE_DIR).mkdir(parents=True, exist_ok=True)

    print(f"\n📁 数据集路径：")
    print(f"  训练图片：{Config.TRAIN_IMG_DIR}")
    print(f"  训练 Mask: {Config.TRAIN_MASK_DIR}")
    print(f"  测试图片：{Config.TEST_IMG_DIR}")
    print(f"  测试 Mask: {Config.TEST_MASK_DIR}")
    print(f"\n💾 结果保存：{Config.SAVE_DIR}")
    print(f"\n🖥 使用设备：{Config.DEVICE}")
    print(f"\n📏 输入尺寸：{Config.IMG_SIZE}")
    print(f"\n⚖️ 损失权重：BCE:Dice = 1:3")
    print(f"\n📊 归一化参数：mean={Config.NORM_MEAN}, std={Config.NORM_STD}")
    print(f"\n➡️ 平移增强：范围 {Config.SHIFT_RANGE} 像素 + 随机水平翻转")

    # 数据集
    print("\n" + "-" * 70)
    print("加载数据集...")
    train_dataset = SegmentationDataset(
        Config.TRAIN_IMG_DIR,
        Config.TRAIN_MASK_DIR,
        img_size=Config.IMG_SIZE,
        is_train=True
    )
    val_dataset = SegmentationDataset(
        Config.TEST_IMG_DIR,
        Config.TEST_MASK_DIR,
        img_size=Config.IMG_SIZE,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKER,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKER,
        pin_memory=True
    )

    print(f"  训练集样本数：{len(train_dataset)}")
    print(f"  验证集样本数：{len(val_dataset)}")

    # 模型
    print("\n" + "-" * 70)
    print("创建模型...")
    model = UNet_ECA(in_channels=3, out_channels=1).to(Config.DEVICE)

    if torch.cuda.is_available() and len(Config.GPU_IDS) > 1:
        available_gpu_count = torch.cuda.device_count()
        valid_gpu_ids = [gid for gid in Config.GPU_IDS if gid < available_gpu_count]

        if len(valid_gpu_ids) > 1:
            print(f"📌 检测到 {available_gpu_count} 块GPU，启用DataParallel多卡训练，使用GPU: {valid_gpu_ids}")
            model = torch.nn.DataParallel(model, device_ids=valid_gpu_ids)
        else:
            print(f"⚠️ 可用GPU不足以启用多卡，当前使用单卡：{Config.DEVICE}")
    else:
        print("📌 使用单卡/CPU训练")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量：{total_params:,}")
    print(f"  可训练参数：{trainable_params:,}")

    # 损失函数 / 优化器 / 调度器
    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS,
        eta_min=1e-6
    )

    # 训练状态初始化
    start_epoch = 1
    history = {
        'train_loss': [], 'train_iou': [], 'train_dice': [],
        'val_loss': [], 'val_iou': [], 'val_dice': []
    }
    best_val_iou = 0.0
    best_epoch = 0

    # 断点恢复
    if Config.RESUME and os.path.exists(Config.CHECKPOINT_PATH):
        start_epoch, history, best_val_iou, best_epoch = load_checkpoint(
            Config.CHECKPOINT_PATH,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=Config.DEVICE
        )
    else:
        print("▶️ 从头开始训练")

    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(start_epoch, Config.EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_iou, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )

        val_loss, val_iou, val_dice = validate(
            model, val_loader, criterion, Config.DEVICE
        )

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)

        # 保存最佳模型
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch

            torch.save({
                'epoch': epoch,
                'model_state_dict': get_model_state_dict(model),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_dice': val_dice,
            }, Path(Config.SAVE_DIR) / 'best_model.pth')

            print(f"  ⭐ 保存最佳模型 (Epoch {epoch}, Val IoU: {val_iou:.4f})")

        # 每10轮保存一次完整checkpoint
        if epoch % 10 == 0:
            save_checkpoint(
                Path(Config.SAVE_DIR) / f'checkpoint_epoch_{epoch}.pth',
                model, optimizer, scheduler, epoch, history, best_val_iou, best_epoch
            )

        # 每轮都保存 latest checkpoint
        save_checkpoint(
            Config.CHECKPOINT_PATH,
            model, optimizer, scheduler, epoch, history, best_val_iou, best_epoch
        )

        # 可视化
        if epoch % 20 == 0 or epoch == 1:
            visualize_results(
                model, val_loader, Config.DEVICE,
                Config.SAVE_DIR, epoch, num_samples=5
            )

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:3d}/{Config.EPOCHS} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f} | "
            f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"  总训练时间：{total_time / 3600:.2f} 小时")
    print(f"  最佳 Epoch: {best_epoch}")
    print(f"  最佳 Val IoU: {best_val_iou:.4f}")

    if best_epoch > 0 and best_epoch <= len(history['val_dice']):
        print(f"  最佳 Val Dice: {history['val_dice'][best_epoch - 1]:.4f}")
    else:
        print("  最佳 Val Dice: N/A")

    # 保存训练历史
    with open(Path(Config.SAVE_DIR) / 'training_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    plot_training_history(history, Config.SAVE_DIR)

    # 保存最终模型
    torch.save({
        'epoch': Config.EPOCHS,
        'model_state_dict': get_model_state_dict(model),
        'optimizer_state_dict': optimizer.state_dict(),
    }, Path(Config.SAVE_DIR) / 'final_model.pth')

    print(f"\n💾 所有结果已保存到：{Config.SAVE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()