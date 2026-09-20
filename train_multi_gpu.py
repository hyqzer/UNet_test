import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import time
from datetime import datetime
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random
from model.unet import *
from model.unet_eca import *


# ==================== 配置部分 ====================
class Config:
    # 数据集路径（不变）
    TRAIN_IMG_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/qiangwei/train_images'
    TRAIN_MASK_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/qiangwei/train_masks'
    TEST_IMG_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/qiangwei/test_images'
    TEST_MASK_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/qiangwei/test_masks'

    # 训练参数（仅修改DEVICE，其余不变）
    EPOCHS = 130
    BATCH_SIZE = 12  # 多卡可适当增大，比如单卡12 → 2卡24/32，根据显存调整
    LEARNING_RATE = 1e-4
    IMG_SIZE = (384, 672)
    NUM_CLASSES = 2
    NUM_WORKER = 6
    # 【修改1】多卡用cuda:0作为主卡，DataParallel自动管理其他GPU
    DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    # 新增：指定使用的GPU编号（可选，比如只用0,1卡，注释则用所有可用卡）
    GPU_IDS = [2,3]  # 根据自己的GPU数量修改，如[0,1,2,3]
    
    SAVE_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/result/unet_qiangwei_0804'
    SEED = 64
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    SHIFT_RANGE = (-50, 50)
    RESUME = False  # True=续训，False=从头开始
    CHECKPOINT_PATH = "/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/result/unet_1_public_0525/checkpoint_epoch_70.pth"

# 补充安全resize函数
def resize_image_and_mask(image, mask, img_size):
    """安全调整图像和掩码尺寸，保持同步"""
    h, w = img_size
    image = TF.resize(image, (h, w))
    mask = TF.resize(mask, (h, w), interpolation=Image.NEAREST)
    return image, mask

# 新增：随机平移增强函数（语义分割专用，图片+掩码严格同步）
def random_shift(image, mask, shift_range):
    """
    对PIL格式的图片和掩码执行随机上下/左右平移
    :param image: PIL.Image 输入图片
    :param mask: PIL.Image 掩码图片
    :param shift_range: tuple (min, max) 平移像素范围
    :return: 平移后的image, mask（PIL格式）
    """
    # 为每张样本生成独立的随机平移量，保证数据集增强的随机性
    shift_x = random.randint(shift_range[0], shift_range[1])
    shift_y = random.randint(shift_range[0], shift_range[1])
    # 图片和掩码执行完全相同的平移操作，填充0（背景）
    image = TF.affine(
        image, angle=0, translate=(shift_x, shift_y),
        scale=1.0, shear=0, fill=0
    )
    mask = TF.affine(
        mask, angle=0, translate=(shift_x, shift_y),
        scale=1.0, shear=0, fill=0  # 掩码填充0，与背景标注逻辑一致
    )
    return image, mask

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(Config.SEED)


# ==================== 数据集类（新增is_train参数+shift增强） ====================

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size, is_train=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.images = sorted(os.listdir(image_dir))
        self.is_train = is_train  # 标记是否为训练集，仅训练集做增强
        # 定义归一化变换（与配置参数绑定）
        self.normalize = transforms.Normalize(
            mean=Config.NORM_MEAN,
            std=Config.NORM_STD
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # 1. 安全resize（图片+掩码同步）
        image, mask = resize_image_and_mask(image, mask, self.img_size)
        
        # 2. 仅训练集执行：随机平移 + 随机水平翻转（双重增强，提升泛化）
        if self.is_train:
            image, mask = random_shift(image, mask, Config.SHIFT_RANGE)
            # 50%概率水平翻转，同样同步处理
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
        
        # 3. 转Tensor + 归一化（保持原有逻辑）
        image = TF.to_tensor(image)
        image = self.normalize(image)
        # mask处理逻辑不变
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
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.25, dice_weight=0.75):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target.float()) + \
               self.dice_weight * self.dice(pred, target.float())

def calculate_iou(pred, target, threshold=0.7):
    """计算 IoU"""
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def calculate_dice(pred, target, threshold=0.5):
    """计算 Dice 系数"""
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    return dice.item()

# ==================== 训练和验证函数 ====================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)  # shape: (N, H, W)
        
        optimizer.zero_grad()
        outputs = model(images)  # shape: (N, 1, H, W)
        
        # 修复：尺寸对齐
        if outputs.shape[2:] != masks.shape[1:]:
            outputs = nn.functional.interpolate(
                outputs, 
                size=(masks.shape[1], masks.shape[2]),  # (H, W)
                mode='bilinear', 
                align_corners=False
            )
        
        loss = criterion(outputs, masks.unsqueeze(1))  # masks: (N, 1, H, W)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_iou += calculate_iou(outputs, masks.unsqueeze(1))
        total_dice += calculate_dice(outputs, masks.unsqueeze(1))
    
    n_batches = len(dataloader)
    return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # 修复：尺寸对齐
            if outputs.shape[2:] != masks.shape[1:]:
                outputs = nn.functional.interpolate(
                    outputs, 
                    size=(masks.shape[1], masks.shape[2]),  # (H, W)
                    mode='bilinear', 
                    align_corners=False
                )
            
            loss = criterion(outputs, masks.unsqueeze(1))
            
            total_loss += loss.item()
            total_iou += calculate_iou(outputs, masks.unsqueeze(1))
            total_dice += calculate_dice(outputs, masks.unsqueeze(1))
    
    n_batches = len(dataloader)
    return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches

# ==================== 可视化函数（适配归一化，还原原图显示） ====================
def visualize_results(model, dataloader, device, save_dir, epoch, num_samples=5):
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    # 反归一化，用于可视化原图（恢复RGB正常显示）
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(Config.NORM_MEAN, Config.NORM_STD)],
        std=[1/s for s in Config.NORM_STD]
    )
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            images = images.to(device)
            outputs = model(images)
            
            if outputs.shape[2:] != masks.shape[1:]:
                outputs = nn.functional.interpolate(outputs, size=masks.shape[1:], 
                                                   mode='bilinear', align_corners=False)
            
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            
            # 原图：先反归一化，再转格式显示
            img = inv_normalize(images[0]).cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)  # 限制像素范围，避免显示异常
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'原图\nSample {i+1}')
            axes[i, 0].axis('off')
            
            # 真实 mask
            axes[i, 1].imshow(masks[0].numpy(), cmap='gray')
            axes[i, 1].set_title('真实 Mask')
            axes[i, 1].axis('off')
            
            # 预测 mask
            axes[i, 2].imshow(preds[0, 0], cmap='gray')
            axes[i, 2].set_title('预测 Mask')
            axes[i, 2].axis('off')
            
            # 叠加效果
            overlay = img.copy()
            overlay[preds[0, 0] > 0.5] = [0, 1, 0]  # 绿色表示预测区域
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('叠加效果')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    save_path = Path(save_dir) / f'epoch_{epoch}_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  可视化结果已保存：{save_path}")

def plot_training_history(history, save_dir):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 损失曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # IoU 曲线
    axes[1].plot(epochs, history['train_iou'], 'b-', label='Train IoU')
    axes[1].plot(epochs, history['val_iou'], 'r-', label='Val IoU')
    axes[1].set_title('IoU')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].legend()
    axes[1].grid(True)
    
    # Dice 曲线
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

# ==================== 主训练函数（修改数据集创建，指定is_train） ====================
def main():
    print("=" * 70)
    print("UNet 分割训练脚本（BCE:Dice=1:3 + 归一化 + Shift+Flip增强）")
    print("=" * 70)
    
    # 创建保存目录
    Path(Config.SAVE_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 数据集路径：")
    print(f"  训练图片：{Config.TRAIN_IMG_DIR}")
    print(f"  训练 Mask: {Config.TRAIN_MASK_DIR}")
    print(f"  测试图片：{Config.TEST_IMG_DIR}")
    print(f"  测试 Mask: {Config.TEST_MASK_DIR}")
    print(f"\n💾 结果保存：{Config.SAVE_DIR}")
    print(f"\n🖥 使用设备：{Config.DEVICE}")
    print(f"\n📏 输入尺寸：{Config.IMG_SIZE}")
    print(f"\n⚖️  损失权重：BCE:Dice = 1:3")
    print(f"\n📊 归一化参数：mean={Config.NORM_MEAN}, std={Config.NORM_STD}")
    print(f"\n➡️  平移增强：范围 {Config.SHIFT_RANGE} 像素 + 随机水平翻转")
    
    # 创建数据集：训练集is_train=True（开启增强），验证集is_train=False（关闭增强）
    print("\n" + "-" * 70)
    print("加载数据集...")
    train_dataset = SegmentationDataset(
        Config.TRAIN_IMG_DIR, 
        Config.TRAIN_MASK_DIR, 
        img_size=Config.IMG_SIZE,
        is_train=True  # 训练集开启shift+flip增强
    )
    
    val_dataset = SegmentationDataset(
        Config.TEST_IMG_DIR, 
        Config.TEST_MASK_DIR, 
        img_size=Config.IMG_SIZE,
        is_train=False  # 验证集关闭所有增强
    )
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                             shuffle=True, num_workers=Config.NUM_WORKER, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, num_workers=Config.NUM_WORKER, pin_memory=True)
    print(f"  训练集样本数：{len(train_dataset)}")
    print(f"  验证集样本数：{len(val_dataset)}")
    
    # 创建模型
    print("\n" + "-" * 70)
    print("创建模型...")
    model = UNet_ECA(in_channels=3, out_channels=1).to(Config.DEVICE)
    
    if torch.cuda.device_count() > 1:
        print(f"📌 检测到 {torch.cuda.device_count()} 块GPU，启用DataParallel多卡训练")
        # 方式1：使用指定的GPU（推荐，可控）
        model = torch.nn.DataParallel(model, device_ids=Config.GPU_IDS)
        # 方式2：使用所有可用GPU（注释上面，打开下面）
        # model = torch.nn.DataParallel(model)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量：{total_params:,}")
    print(f"  可训练参数：{trainable_params:,}")
    
    # 损失函数和优化器
    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    # 原来这个是ReduceLROnPlateau
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
    #                                                   factor=0.5, patience=5, verbose=True)
    # ========== 新增：余弦退火学习率调度（推荐，适配小数据集） ==========
    # T_max=训练总轮数，eta_min=最小学习率（避免降为0）
    # 学习率调度（只保留一次）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS,
        eta_min=1e-6,
        verbose=True
    )

    # ========== 断点续训（正确版，只初始化一次） ==========
    start_epoch = 1
    history = {
        'train_loss': [], 'train_iou': [], 'train_dice': [],
        'val_loss': [], 'val_iou': [], 'val_dice': []
    }
    best_val_iou = 0
    best_epoch = 0

    if Config.RESUME and os.path.exists(Config.CHECKPOINT_PATH):
        print(f"🔄 从断点恢复：{Config.CHECKPOINT_PATH}")
        checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
        
        # ========== 多卡修复核心代码 ==========
        state_dict = checkpoint['model_state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict['module.' + k] = v
        model.load_state_dict(new_state_dict)
        # ====================================
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        best_val_iou = checkpoint['best_val_iou']
        best_epoch = checkpoint['best_epoch']
        print(f"✅ 恢复成功，从 Epoch {start_epoch} 继续训练")
    else:
        print("▶️ 从头开始训练")

    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, Config.EPOCHS + 1):
        epoch_start = time.time()
        
        # 训练
        train_loss, train_iou, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        
        # 验证
        val_loss, val_iou, val_dice = validate(
            model, val_loader, criterion, Config.DEVICE
        )
        
        # ========== 若用回原ReduceLROnPlateau，需传val_loss（注释上面，打开下面） ==========
        # scheduler.step(val_loss)
        # ========== 余弦退火调度：每个epoch直接step（无需传val_loss） ==========
        scheduler.step()
        
        # 记录历史
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
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_dice': val_dice,
            }, Path(Config.SAVE_DIR) / 'best_model.pth')
            print(f"  ⭐ 保存最佳模型 (Epoch {epoch}, Val IoU: {val_iou:.4f})")
        
        # 每 10 个 epoch 保存一次检查点
        if epoch % 10 == 0:
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
            }, Path(Config.SAVE_DIR) / f'checkpoint_epoch_{epoch}.pth')
        
        # 每 20 个 epoch 可视化一次
        if epoch % 20 == 0 or epoch == 1:
            visualize_results(model, val_loader, Config.DEVICE, 
                            Config.SAVE_DIR, epoch, num_samples=5)
        
        epoch_time = time.time() - epoch_start
        
        # 打印epoch信息，新增lr打印
        current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
        print(f"Epoch {epoch:3d}/{Config.EPOCHS} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f} | "
            f"Time: {epoch_time:.1f}s")

        # ===================== 这里是正确位置 =====================
        # 保存最新断点（用于续训）
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_val_iou': best_val_iou,
            'best_epoch': best_epoch
        }
        torch.save(checkpoint, Config.CHECKPOINT_PATH)
        # ==========================================================

    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"  总训练时间：{total_time/3600:.2f} 小时")
    print(f"  最佳 Epoch: {best_epoch}")
    print(f"  最佳 Val IoU: {best_val_iou:.4f}")
    print(f"  最佳 Val Dice: {history['val_dice'][best_epoch-1]:.4f}")
    
    # 保存训练历史
    import json
    with open(Path(Config.SAVE_DIR) / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # 绘制训练曲线
    plot_training_history(history, Config.SAVE_DIR)
    
    # 保存最终模型
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({
        'epoch': Config.EPOCHS,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
    }, Path(Config.SAVE_DIR) / 'final_model.pth')
    
    print(f"\n💾 所有结果已保存到：{Config.SAVE_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()