import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime

# ==================== 配置部分 ====================

class Config:
    # 模型路径
    MODEL_PATH = '/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/unet_results/best_model.pth'
    
    # 测试集路径
    TEST_IMG_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/test_image'
    TEST_MASK_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/test_mask'
    
    # 输出路径
    SAVE_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/inference_results2'
    
    # 推理参数
    IMG_SIZE = 256
    BATCH_SIZE = 4
    DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    THRESHOLD = 0.9  # 二值化阈值
    
    # 随机种子
    SEED = 42

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(Config.SEED)

# ==================== UNet 模型定义（与训练时一致）====================

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
        
        # 编码器
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # 瓶颈层
        self.bottleneck = DoubleConv(512, 1024)
        
        # 解码器
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # 输出层
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # 瓶颈层
        b = self.bottleneck(self.pool(e4))
        
        # 解码器
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out_conv(d1)

# ==================== 数据集类 ====================

class InferenceDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, img_size=256, has_mask=True):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.img_size = img_size
        self.has_mask = has_mask
        
        # 获取图片文件列表
        self.img_files = sorted([f for f in self.img_dir.glob('*.png') 
                                if not f.name.startswith('.')])
        self.img_files += sorted([f for f in self.img_dir.glob('*.jpg')
                                 if not f.name.startswith('.')])
        
        print(f"  找到 {len(self.img_files)} 张测试图片")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img_name = img_path.stem
        
        # 加载图片
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # 保存原始尺寸
        
        # 图片变换
        img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        image = img_transform(image)
        
        # 加载 mask（如果有）
        mask = None
        if self.has_mask and self.mask_dir:
            mask_path = self.mask_dir / f"{img_name}.png"
            if mask_path.exists():
                mask = Image.open(mask_path).convert('L')
                mask_transform = transforms.Compose([
                    transforms.Resize((self.img_size, self.img_size), 
                                    interpolation=Image.NEAREST),
                ])
                mask = mask_transform(mask)
                mask = torch.from_numpy(np.array(mask)).long()
                mask = (mask > 0).long()  # 确保只有 0 和 1
        
        return image, mask, img_name, original_size

# ==================== 评估指标 ====================

def calculate_iou(pred, target):
    """计算 IoU"""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def calculate_dice(pred, target):
    """计算 Dice 系数"""
    intersection = (pred * target).sum()
    dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    return dice.item()

def calculate_metrics(pred_mask, true_mask):
    """计算所有评估指标"""
    pred = pred_mask.flatten()
    target = true_mask.flatten()
    
    # TP, FP, TN, FN
    TP = ((pred == 1) & (target == 1)).sum()
    FP = ((pred == 1) & (target == 0)).sum()
    TN = ((pred == 0) & (target == 0)).sum()
    FN = ((pred == 0) & (target == 1)).sum()
    
    # 指标计算
    iou = (TP + 1e-6) / (TP + FP + FN + 1e-6)
    dice = (2 * TP + 1e-6) / (2 * TP + FP + FN + 1e-6)
    precision = (TP + 1e-6) / (TP + FP + 1e-6)
    recall = (TP + 1e-6) / (TP + FN + 1e-6)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    
    # ✅ 关键修复：全部转换为 Python 原生类型 (float 或 int)
    return {
        'IoU': float(iou),
        'Dice': float(dice),
        'Precision': float(precision),
        'Recall': float(recall),
        'Accuracy': float(accuracy),
        'TP': int(TP),      # 从 Tensor 转为 int
        'FP': int(FP),      # 从 Tensor 转为 int
        'TN': int(TN),      # 从 Tensor 转为 int
        'FN': int(FN)       # 从 Tensor 转为 int
    }

# ==================== 推理函数 ====================

def load_model(model_path, device):
    """加载训练好的模型"""
    print(f"\n📦 加载模型：{model_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在：{model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  ✓ 模型加载成功")
    print(f"  训练 Epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'val_iou' in checkpoint:
        print(f"  验证 IoU: {checkpoint['val_iou']:.4f}")
    if 'val_dice' in checkpoint:
        print(f"  验证 Dice: {checkpoint['val_dice']:.4f}")
    
    return model

def predict_single(model, image, device, img_size, threshold=0.5):
    """单张图片预测"""
    image = image.unsqueeze(0).to(device)  # (1, 3, H, W)
    
    with torch.no_grad():
        output = model(image)
        output = nn.functional.interpolate(
            output, 
            size=(img_size, img_size),
            mode='bilinear', 
            align_corners=False
        )
        pred = (torch.sigmoid(output) > threshold).cpu().numpy()[0, 0]
    
    return pred

def visualize_results(image, true_mask, pred_mask, img_name, save_path):
    """可视化单张图片的预测结果"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 原图（去归一化）
    img = image.numpy().transpose(1, 2, 0)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    axes[0].imshow(img)
    axes[0].set_title(f'原图\n{img_name}')
    axes[0].axis('off')
    
    # 真实 Mask
    if true_mask is not None:
        axes[1].imshow(true_mask.numpy(), cmap='gray')
        axes[1].set_title('真实 Mask')
    else:
        axes[1].imshow(np.zeros((256, 256)), cmap='gray')
        axes[1].set_title('真实 Mask (无)')
    axes[1].axis('off')
    
    # 预测 Mask
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title(f'预测 Mask')
    axes[2].axis('off')
    
    # 叠加效果
    overlay = img.copy()
    overlay[pred_mask > 0.5] = [0, 1, 0]  # 绿色表示预测区域
    axes[3].imshow(overlay)
    axes[3].set_title('叠加效果 (绿色=预测)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def inference(model, dataloader, device, save_dir, threshold=0.5):
    """批量推理"""
    print("\n" + "=" * 70)
    print("开始推理...")
    print("=" * 70)
    
    # 创建输出目录
    pred_dir = Path(save_dir) / 'predictions'
    viz_dir = Path(save_dir) / 'visualizations'
    pred_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    all_metrics = []
    
    start_time = time.time()
    
    for idx, (images, masks, names, original_sizes) in enumerate(dataloader):
        batch_start = time.time()
        
        for i in range(len(images)):
            image = images[i]
            mask = masks[i] if masks is not None else None
            img_name = names[i]
            orig_size = original_sizes[i]
            
            # 预测
            pred_mask = predict_single(model, image, device, Config.IMG_SIZE, threshold)
            
            # 保存预测结果（二值 PNG）
            pred_img = (pred_mask * 255).astype(np.uint8)
            pred_pil = Image.fromarray(pred_img)
            pred_pil.save(pred_dir / f'{img_name}_pred.png')
            
            # 如果有真实 mask，计算指标
            sample_metrics = {'name': img_name}
            if mask is not None:
                metrics = calculate_metrics(torch.tensor(pred_mask), mask)
                sample_metrics.update(metrics)
                all_metrics.append(metrics)
                
                # 可视化
                viz_path = viz_dir / f'{img_name}_result.png'
                visualize_results(image, mask, pred_mask, img_name, viz_path)
            else:
                # 无真实 mask，只保存预测可视化
                viz_path = viz_dir / f'{img_name}_result.png'
                visualize_results(image, None, pred_mask, img_name, viz_path)
            
            results.append(sample_metrics)
        
        batch_time = time.time() - batch_start
        print(f"  批次 {idx+1}/{len(dataloader)} | 耗时：{batch_time:.2f}s")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("推理完成！")
    print("=" * 70)
    print(f"  总耗时：{total_time:.2f}s")
    print(f"  平均速度：{len(dataloader.dataset)/total_time:.2f} 图片/秒")
    print(f"  预测结果：{pred_dir}")
    print(f"  可视化结果：{viz_dir}")
    
    return results, all_metrics

def save_results(results, all_metrics, save_dir):
    """保存评估结果"""
    # 保存每个样本的结果
    with open(Path(save_dir) / 'inference_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 如果有真实 mask，计算平均指标
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key not in ['name', 'TP', 'FP', 'TN', 'FN']:
                avg_metrics[f'Average_{key}'] = np.mean([m[key] for m in all_metrics])
            else:
                avg_metrics[f'Total_{key}'] = sum([m[key] for m in all_metrics])
        
        # 保存平均指标
        with open(Path(save_dir) / 'average_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(avg_metrics, f, indent=2, ensure_ascii=False)
        
        # 打印平均指标
        print("\n" + "=" * 70)
        print("📊 评估指标（平均）")
        print("=" * 70)
        for key, value in avg_metrics.items():
            print(f"  {key}: {value:.4f}")
        print("=" * 70)
    
    print(f"\n💾 结果已保存：{save_dir}")

# ==================== 主函数 ====================

def main():
    print("=" * 70)
    print("UNet 推理脚本")
    print("=" * 70)
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    Config.SAVE_DIR = f"{Config.SAVE_DIR}_{timestamp}"
    Path(Config.SAVE_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 配置信息：")
    print(f"  模型路径：{Config.MODEL_PATH}")
    print(f"  测试图片：{Config.TEST_IMG_DIR}")
    print(f"  测试 Mask: {Config.TEST_MASK_DIR}")
    print(f"  结果保存：{Config.SAVE_DIR}")
    print(f"  使用设备：{Config.DEVICE}")
    print(f"  阈值：{Config.THRESHOLD}")
    
    # 检查测试集是否有 mask
    has_mask = Path(Config.TEST_MASK_DIR).exists() and len(list(Path(Config.TEST_MASK_DIR).glob('*.png'))) > 0
    print(f"  有真实 Mask: {has_mask}")
    
    # 加载模型
    model = load_model(Config.MODEL_PATH, Config.DEVICE)
    
    # 创建数据集
    print("\n" + "-" * 70)
    print("加载测试集...")
    dataset = InferenceDataset(
        Config.TEST_IMG_DIR,
        Config.TEST_MASK_DIR if has_mask else None,
        img_size=Config.IMG_SIZE,
        has_mask=has_mask
    )
    
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # 开始推理
    results, all_metrics = inference(
        model, 
        dataloader, 
        Config.DEVICE, 
        Config.SAVE_DIR,
        Config.THRESHOLD
    )
    
    # 保存结果
    save_results(results, all_metrics, Config.SAVE_DIR)
    
    print("\n✅ 所有任务完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()