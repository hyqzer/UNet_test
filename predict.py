import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import torchvision.transforms.functional as TF

# ==================== 1. 模型架构 (与训练代码完全一致) ====================
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

# ==================== 2. 后处理工具 (过滤小噪点，保留最大连通域) ====================
def keep_largest_connected_component(mask):
    """
    只保留二值Mask中面积最大的连通区域，过滤小噪声
    mask: 0/255的单通道numpy数组
    """
    mask = mask.astype(np.uint8)
    # 寻找8连通域（更贴合实际分割的连通性）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:  # 只有背景，无目标区域
        return mask
    # 找到除背景外面积最大的连通域
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    # 生成新的二值Mask
    new_mask = np.zeros_like(mask)
    new_mask[labels == largest_label] = 255
    return new_mask

# ==================== 3. 推理配置 (与训练配置严格对齐，需根据实际修改) ====================
class InferConfig:
    # 训练好的模型权重路径 (best_model.pth/final_model.pth)
    MODEL_PATH = '/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/unet_0304_light/best_model.pth'
    # 待推理图片目录/单张图片路径
    INPUT_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/predict'
    # 推理结果保存目录（自动创建）
    OUTPUT_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/prediction/2'
    # 与训练一致的输入尺寸 (H, W) - 固定384×672
    IMG_SIZE = (384, 672)
    # 二值化阈值 (与训练时一致，默认0.5)
    THRESHOLD = 0.05
    # 推理设备 (与训练一致用cuda:1，无则自动切cpu)
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # 是否保留最大连通域 (True=过滤小噪点，False=原始预测)
    ONLY_KEEP_LARGEST = True
    # 可视化叠加透明度 (0~1，越小原图越清晰)
    OVERLAY_ALPHA = 0.3
    # 归一化参数 (与训练Config完全一致，不可修改)
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

# ==================== 4. 推理核心函数 ====================
def run_inference():
    # 创建输出目录（不存在则自动创建）
    out_path = Path(InferConfig.OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 推理结果将保存至：{out_path.absolute()}")

    # 加载模型 (与训练代码完全一致)
    model = UNet(in_channels=3, out_channels=1).to(InferConfig.DEVICE)
    # 加载权重 (兼容训练的checkpoint格式，只取模型权重)
    checkpoint = torch.load(InferConfig.MODEL_PATH, map_location=InferConfig.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 切换推理模式，关闭BN/Dropout
    print(f"✅ 模型权重已从 {InferConfig.MODEL_PATH} 加载")
    print(f"🖥 推理使用设备：{InferConfig.DEVICE}")
    print(f"⚙️  推理配置：输入尺寸{InferConfig.IMG_SIZE} | 二值化阈值{InferConfig.THRESHOLD} | 归一化{InferConfig.NORM_MEAN}/{InferConfig.NORM_STD}")

    # 预处理 (与训练代码严格一致：Resize+ToTensor+Normalize)
    def preprocess(image):
        # 调整尺寸到训练的384×672
        image = TF.resize(image, InferConfig.IMG_SIZE)
        # 转Tensor (将0-255转为0-1)
        image = TF.to_tensor(image)
        # 归一化 (与训练完全一致，核心！)
        image = TF.normalize(image, mean=InferConfig.NORM_MEAN, std=InferConfig.NORM_STD)
        # 增加batch维度 (模型输入要求[B, C, H, W])
        image = image.unsqueeze(0).to(InferConfig.DEVICE)
        return image

    # 获取待推理图片列表 (支持单张图片/图片目录，自动过滤格式)
    img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']
    image_files = []
    input_path = Path(InferConfig.INPUT_DIR)
    if input_path.is_file() and input_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
        image_files = [input_path]
    elif input_path.is_dir():
        for ext in img_extensions:
            image_files.extend(sorted(input_path.glob(ext)))
    else:
        raise FileNotFoundError(f"❌ 输入路径无效，请检查：{InferConfig.INPUT_DIR}")
    
    if len(image_files) == 0:
        print("❌ 未找到待推理的图片文件（支持png/jpg/jpeg/bmp/tif）")
        return
    print(f"🔍 找到 {len(image_files)} 张待推理图片，开始推理...")

    # 批量推理 (关闭梯度计算，提升速度并节省显存)
    with torch.no_grad():
        for idx, img_path in enumerate(image_files, 1):
            try:
                # 1. 加载原始图片并记录原图尺寸 (保留PIL格式，避免cv2色域问题)
                original_img = Image.open(img_path).convert("RGB")
                orig_h, orig_w = original_img.size[1], original_img.size[0]  # (H, W)
                # 2. 预处理 (严格匹配训练流程)
                input_tensor = preprocess(original_img)
                # 3. 模型推理：输出384×672的logits
                output = model(input_tensor)
                # 4. 核心：在384×672分辨率下完成sigmoid+二值化 (避免插值模糊)
                pred = torch.sigmoid(output).cpu().squeeze().numpy()  # 去掉batch和channel维度
                mask_384_672 = (pred > InferConfig.THRESHOLD).astype(np.uint8) * 255  # 转0/255二值图
                
                # 5. 核心：最近邻插值将mask放大回原图尺寸 (保留二值边缘，无模糊)
                # cv2.resize的dsize参数为(w, h)，与orig_w/orig_h对应
                mask_original = cv2.resize(
                    mask_384_672,
                    dsize=(orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST  # 最近邻插值，必选！
                )

                # 6. 可选后处理：过滤小噪点，保留最大连通域
                if InferConfig.ONLY_KEEP_LARGEST:
                    mask_original = keep_largest_connected_component(mask_original)

                # 7. 保存推理结果
                # 保存纯二值Mask (原图尺寸，可直接用于后续计算)
                mask_img = Image.fromarray(mask_original)
                mask_img.save(out_path / f"mask_{img_path.stem}.png")
                # 保存叠加可视化结果 (绿色预测区域，直观查看效果)
                img_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)  # 转cv2格式
                colored_mask = np.zeros_like(img_cv)
                colored_mask[mask_original > 0] = [0, 255, 0]  # 预测区域标为绿色
                overlay = cv2.addWeighted(
                    img_cv, 1 - InferConfig.OVERLAY_ALPHA,
                    colored_mask, InferConfig.OVERLAY_ALPHA, 0
                )
                cv2.imwrite(out_path / f"overlay_{img_path.stem}.jpg", overlay)

                print(f"  [{idx}/{len(image_files)}] ✅ 处理完成：{img_path.name}")
            except Exception as e:
                # 单张图片失败不中断整体推理，打印错误信息
                print(f"  [{idx}/{len(image_files)}] ❌ 处理失败：{img_path.name} | 错误：{str(e)}")

    print(f"\n✨ 推理全部完成！所有结果已保存至：{out_path.absolute()}")
    print(f"📌 输出文件说明：mask_xxx.png=纯二值分割结果 | overlay_xxx.jpg=原图+预测叠加可视化")

# ==================== 执行推理 ====================
if __name__ == "__main__":
    print("=" * 80)
    print("UNet 语义分割推理脚本 (适配训练配置：384×672+归一化+BCE:Dice=1:3)")
    print("=" * 80)
    run_inference()