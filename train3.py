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

# ==================== 配置部分 ====================
class Config:
    # 数据集路径
    TRAIN_IMG_DIR = '/home/ytkj/work/hangyq/Projects/ultralytics-main/shitou/images/train'
    TRAIN_MASK_DIR = '/home/ytkj/work/hangyq/Projects/ultralytics-main/shitou/masks/train'
    TEST_IMG_DIR = '/home/ytkj/work/hangyq/Projects/ultralytics-main/shitou/images/test'
    TEST_MASK_DIR = '/home/ytkj/work/hangyq/Projects/ultralytics-main/shitou/masks/test'
    
    # 训练参数
    EPOCHS = 200
    BATCH_SIZE = 12
    LEARNING_RATE = 1e-5 #1e-4
    IMG_SIZE = (384, 672)  # 输入图片尺寸 (H, W)   
    NUM_CLASSES = 2  # 背景 + water
    NUM_WORKER = 4
    DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    
    # 保存路径
    SAVE_DIR = '/home/ytkj/work/hangyq/Projects/ultralytics-main/shitou/unet'
    
    # 随机种子Q
    SEED = 32
    # 新增：归一化参数（ImageNet标准）
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    # 平移增强配置
    SHIFT_RANGE = (-0, 0)  # 上下左右平移像素范围，可根据需求调整


    # ===== 新增：光照增强参数（核心解决光照不均）=====
    BRIGHTNESS_RANGE = (1.0, 1.0)  # 亮度因子，0.5=暗一倍，1.5=亮一倍
    CONTRAST_RANGE = (0.5, 1.5)    # 对比度因子
    GAMMA_RANGE = (0.7, 1.3)       # 伽马矫正，<1变亮，>1变暗
    HUE_RANGE = (-0.1, 0.1)        # 色相微调，范围越小越不改变原始色彩
    SATURATION_RANGE = (0.8, 1.2)  # 饱和度微调
    # ===== 新增：几何增强参数（可选）=====
    ROTATE_RANGE = (-15, 15)       # 随机旋转角度（小角度，避免目标变形）
    SCALE_RANGE = (0.8, 1.2)       # 随机缩放因子

    SCALE_RANGE = (0.8, 1.2)       # 随机缩放因子
    
    # ===== 新增：CLAHE 配置（光照预处理）=====
    CLAHE_CLIP_LIMIT = 2.0         # CLAHE对比度限制，越小越柔和（推荐2.0）
    CLAHE_GRID_SIZE = (8, 8)       # CLAHE网格大小，默认8x8
    
    # ===== 新增：边界检测（Sobel）配置 =====
    SOBEL_PROB = 0.5               # 训练集随机融合Sobel边缘的概率（50%）
    SOBEL_ALPHA = 0.3              # 边缘特征的融合权重（原图0.7+边缘0.3）


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


# ===== 新增：随机光照增强（仅处理图片，mask不变）=====
def random_light_aug(image):
    """
    对PIL图片执行光照相关增强，解决光照不均
    :param image: PIL.Image 输入图片（RGB）
    :return: 增强后的PIL图片
    """
    # 1. 随机亮度调整
    brightness = random.uniform(*Config.BRIGHTNESS_RANGE)
    image = TF.adjust_brightness(image, brightness)
    
    # 2. 随机对比度调整
    contrast = random.uniform(*Config.CONTRAST_RANGE)
    image = TF.adjust_contrast(image, contrast)
    
    # 3. 随机伽马矫正（非线性亮度调整，核心解决光照不均）
    gamma = random.uniform(*Config.GAMMA_RANGE)
    image = TF.adjust_gamma(image, gamma)
    
    # 4. 随机色相微调（可选，注释则关闭）
    hue = random.uniform(*Config.HUE_RANGE)
    image = TF.adjust_hue(image, hue)
    
    # 5. 随机饱和度微调（可选，注释则关闭）
    saturation = random.uniform(*Config.SATURATION_RANGE)
    image = TF.adjust_saturation(image, saturation)
    
    # 6. 随机轻微高斯模糊（可选，降低光照噪点，注释则关闭）
    if random.random() > 0.7:  # 70%概率不模糊，30%概率模糊
        image = TF.gaussian_blur(image, kernel_size=(3, 3), sigma=(0.1, 2.0))
    
    return image

def clahe_preprocess(image):
    """
    对PIL RGB图像执行CLAHE预处理，解决光照不均
    :param image: PIL.Image (RGB) 输入图片
    :return: PIL.Image (RGB) CLAHE处理后的图片
    """
    import cv2
    import numpy as np
    # PIL转OpenCV（BGR格式）
    img_cv = np.array(image)[:, :, ::-1]
    # 转YCrCb色彩空间（仅对亮度通道做CLAHE，保留色彩）
    ycrcb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    # 初始化CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=Config.CLAHE_CLIP_LIMIT,
        tileGridSize=Config.CLAHE_GRID_SIZE
    )
    y_clahe = clahe.apply(y)
    # 合并通道并转回RGB
    ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
    img_clahe = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)
    img_clahe = img_clahe[:, :, ::-1]
    # OpenCV转回PIL
    return Image.fromarray(img_clahe)

# ===== 新增：Sobel边界检测+融合函数（仅训练集随机执行）=====
def sobel_edge_fusion(image):
    """
    对PIL RGB图像提取Sobel边缘，与原图融合增强边界特征
    :param image: PIL.Image (RGB) 输入图片
    :return: PIL.Image (RGB) 融合边缘后的图片
    """
    import cv2
    import numpy as np
    # PIL转OpenCV（灰度图，边缘检测用灰度图即可）
    img_gray = np.array(image.convert('L'), dtype=np.float32) / 255.0
    # Sobel水平+垂直边缘检测
    sobel_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    # 融合边缘（绝对值+归一化）
    sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_edges = (sobel_edges - sobel_edges.min()) / (sobel_edges.max() - sobel_edges.min() + 1e-8)
    # 边缘图转3通道（与原图匹配）
    sobel_3ch = np.stack([sobel_edges]*3, axis=-1)
    # 原图转np数组（归一化）
    img_np = np.array(image, dtype=np.float32) / 255.0
    # 融合原图和边缘图（加权叠加）
    fused_img = (1 - Config.SOBEL_ALPHA) * img_np + Config.SOBEL_ALPHA * sobel_3ch
    fused_img = (np.clip(fused_img, 0, 1) * 255).astype(np.uint8)
    # 转回PIL
    return Image.fromarray(fused_img)



# ===== 新增：随机旋转+缩放增强（图片+mask同步，可选）=====
def random_rotate_scale(image, mask, rotate_range, scale_range):
    """
    小角度旋转+随机缩放，图片+mask同步处理，保证标注对齐
    """
    # 随机旋转角度
    angle = random.uniform(*rotate_range)
    # 随机缩放因子
    scale = random.uniform(*scale_range)
    # 旋转+缩放，图片填充0（背景），mask填充0（与标注逻辑一致）
    image = TF.affine(
        image, angle=angle, translate=(0, 0), scale=scale, shear=0, fill=0
    )
    mask = TF.affine(
        mask, angle=angle, translate=(0, 0), scale=scale, shear=0, fill=0,
        interpolation=Image.NEAREST  # mask用最近邻插值，避免标注模糊
    )
    return image, mask


# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(Config.SEED)

# ==================== UNet 模型定义 ====================
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
    def __init__(self, in_channels=3, out_channels=1):
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
        self.out_conv = nn.Conv2d(64, out_channels, 1)
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

# ==================== 数据集类（新增is_train参数+shift增强） ====================
from torch.utils.data import Dataset
import os
import torch

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

        image = clahe_preprocess(image)
        
        # 3. 仅训练集执行：几何增强+光照增强+随机边缘融合
        if self.is_train:
            # 几何增强（图片+mask同步）
            image, mask = random_shift(image, mask, Config.SHIFT_RANGE)
            image, mask = random_rotate_scale(image, mask, Config.ROTATE_RANGE, Config.SCALE_RANGE)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # 光照增强（仅图片）
            image = random_light_aug(image)
            
            # 随机Sobel边界融合（仅图片，按概率执行）
            if random.random() < Config.SOBEL_PROB:
                image = sobel_edge_fusion(image)
        
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
    print(f"\n☀️  CLAHE预处理：clipLimit={Config.CLAHE_CLIP_LIMIT}, gridSize={Config.CLAHE_GRID_SIZE}")
    print(f"\n📏  Sobel边界融合：概率{Config.SOBEL_PROB}, 融合权重{Config.SOBEL_ALPHA}")
    
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
    model = UNet(in_channels=3, out_channels=1).to(Config.DEVICE)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量：{total_params:,}")
    print(f"  可训练参数：{trainable_params:,}")
    
    # 损失函数和优化器
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=5, verbose=True)
    
    # 训练历史记录
    history = {
        'train_loss': [], 'train_iou': [], 'train_dice': [],
        'val_loss': [], 'val_iou': [], 'val_dice': []
    }
    
    best_val_iou = 0
    best_epoch = 0
    
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)
    
    start_time = time.time()
    
    for epoch in range(1, Config.EPOCHS + 1):
        epoch_start = time.time()
        
        # 训练
        train_loss, train_iou, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        
        # 验证
        val_loss, val_iou, val_dice = validate(
            model, val_loader, criterion, Config.DEVICE
        )
        
        # 更新学习率
        scheduler.step(val_loss)
        
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_dice': val_dice,
            }, Path(Config.SAVE_DIR) / 'best_model.pth')
            print(f"  ⭐ 保存最佳模型 (Epoch {epoch}, Val IoU: {val_iou:.4f})")
        
        # 每 10 个 epoch 保存一次检查点
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, Path(Config.SAVE_DIR) / f'checkpoint_epoch_{epoch}.pth')
        
        # 每 20 个 epoch 可视化一次
        if epoch % 20 == 0 or epoch == 1:
            visualize_results(model, val_loader, Config.DEVICE, 
                            Config.SAVE_DIR, epoch, num_samples=5)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch:3d}/{Config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f} | "
              f"Time: {epoch_time:.1f}s")
    
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
    torch.save({
        'epoch': Config.EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, Path(Config.SAVE_DIR) / 'final_model.pth')
    
    print(f"\n💾 所有结果已保存到：{Config.SAVE_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()