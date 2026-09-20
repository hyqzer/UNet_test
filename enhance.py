import os
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import cv2
from pathlib import Path
import random

# ==================== 1. 配置参数 (移除augmented子文件夹配置) ====================
class AugConfig:
    # 原始数据路径 (增强后文件直接保存在这些目录下)
    TRAIN_IMG_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/blur/train_images'
    TRAIN_MASK_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/blur/train_masks'
    TEST_IMG_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/blur/test_images'
    TEST_MASK_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/blur/test_masks'
    
    # 增强参数 (可根据需求调整)
    BRIGHTNESS_FACTORS = [0.5, 0.7, 1.3, 1.5]  # 亮度调整因子：<1降低，>1增强
    CONTRAST_FACTORS = [1.2, 1.5, 1.8]         # 对比度增强因子：>1增强
    GAMMA_VALUES = [0.6, 0.8, 1.2, 1.4]        # Gamma变换值：<1提亮，>1变暗
    AUGMENT_TIMES = 1  # 每张图片生成多少份增强数据 (建议1，避免数据量过大)
    
    # 随机种子 (保证增强结果可复现)
    SEED = 42

# 设置随机种子
random.seed(AugConfig.SEED)
np.random.seed(AugConfig.SEED)

# ==================== 2. 数据增强核心函数 (仅增强图片，mask原样复制) ====================
def adjust_brightness(image, factor):
    """调整图片亮度"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor):
    """增强图片对比度"""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def adjust_gamma(image, gamma=1.0):
    """Gamma变换 (OpenCV实现，更适合像素级调整)"""
    # 将PIL图片转为numpy数组
    img_np = np.array(image).astype(np.float32) / 255.0
    # Gamma变换核心公式
    img_np = np.power(img_np, gamma)
    # 转回0-255 uint8
    img_np = (img_np * 255).astype(np.uint8)
    # 转回PIL图片
    return Image.fromarray(img_np)

def random_augment(image):
    """对单张图片随机应用一种增强方式 (避免过度增强)"""
    aug_type = random.choice(['brightness', 'contrast', 'gamma'])
    
    if aug_type == 'brightness':
        factor = random.choice(AugConfig.BRIGHTNESS_FACTORS)
        aug_img = adjust_brightness(image, factor)
        suffix = f'_bright_{factor}'
    elif aug_type == 'contrast':
        factor = random.choice(AugConfig.CONTRAST_FACTORS)
        aug_img = adjust_contrast(image, factor)
        suffix = f'_contrast_{factor}'
    elif aug_type == 'gamma':
        gamma = random.choice(AugConfig.GAMMA_VALUES)
        aug_img = adjust_gamma(image, gamma)
        suffix = f'_gamma_{gamma}'
    
    return aug_img, suffix

# ==================== 3. 批量增强函数 (增强后文件直接保存到原目录) ====================
def batch_augment(img_dir, mask_dir, is_train=True):
    """
    批量增强图片，mask原样复制，增强后文件直接保存到原目录
    :param img_dir: 原始图片目录 (增强后图片直接保存在此)
    :param mask_dir: 原始mask目录 (增强后mask直接保存在此)
    :param is_train: 是否为训练集 (训练集增强，测试集可选增强)
    """
    # 获取所有图片文件 (仅处理常见格式)
    img_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
    img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in img_extensions]
    
    if len(img_files) == 0:
        print(f"❌ 未在 {img_dir} 找到图片文件")
        return
    
    print(f"\n📌 开始处理 {['训练集' if is_train else '测试集']}，共 {len(img_files)} 张图片")
    print(f"🔧 增强方式：亮度调整 | 对比度增强 | Gamma变换")
    print(f"💾 增强后文件直接保存至原目录：{img_dir} (图片) | {mask_dir} (Mask)")
    
    # 批量处理
    for idx, img_name in enumerate(img_files, 1):
        try:
            # 1. 加载原始图片和mask
            img_path = os.path.join(img_dir, img_name)
            mask_name = img_name  # 假设mask和图片文件名完全一致
            mask_path = os.path.join(mask_dir, mask_name)
            
            if not os.path.exists(mask_path):
                print(f"⚠️  未找到mask文件：{mask_path}，跳过该图片")
                continue
            
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # mask为单通道灰度图
            
            # 2. 生成增强数据 (每张图片生成AUGMENT_TIMES份)
            for aug_idx in range(AugConfig.AUGMENT_TIMES):
                # 随机增强图片
                aug_img, suffix = random_augment(image)
                
                # 3. 构造增强后的文件名 (原名称+增强类型+因子，避免覆盖原始文件)
                img_base, img_ext = os.path.splitext(img_name)
                aug_img_name = f"{img_base}_{aug_idx+1}{suffix}{img_ext}"
                aug_mask_name = f"{img_base}_{aug_idx+1}{suffix}{img_ext}"  # mask文件名和图片对应
                
                # 4. 保存增强后的图片和原样的mask到原目录
                aug_img.save(os.path.join(img_dir, aug_img_name))
                mask.save(os.path.join(mask_dir, aug_mask_name))
            
            if idx % 20 == 0:
                print(f"  ✅ 已处理 {idx}/{len(img_files)} 张图片")
        
        except Exception as e:
            print(f"❌ 处理 {img_name} 失败：{str(e)}")
    
    # 统计增强后的数据量 (原始+增强)
    total_img_count = len([f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in img_extensions])
    total_mask_count = len([f for f in os.listdir(mask_dir) if os.path.splitext(f)[1].lower() in img_extensions])
    aug_img_count = total_img_count - len(img_files)  # 新增的增强图片数
    aug_mask_count = total_mask_count - len(img_files)  # 新增的增强mask数
    
    print(f"\n✅ {['训练集' if is_train else '测试集']} 增强完成！")
    print(f"   原始图片数量：{len(img_files)} | 新增增强图片：{aug_img_count} | 总图片数：{total_img_count}")
    print(f"   原始Mask数量：{len(img_files)} | 新增增强Mask：{aug_mask_count} | 总Mask数：{total_mask_count}")

# ==================== 4. 主函数 (调用修改后的批量增强函数) ====================
def main():
    print("=" * 80)
    print("UNet 分割任务数据增强脚本 (增强后文件直接保存到原目录)")
    print("=" * 80)
    
    # 处理训练集 (必增强)
    batch_augment(
        img_dir=AugConfig.TRAIN_IMG_DIR,
        mask_dir=AugConfig.TRAIN_MASK_DIR,
        is_train=True
    )
    
    # 处理测试集 (可选增强，若不需要可注释)
    batch_augment(
        img_dir=AugConfig.TEST_IMG_DIR,
        mask_dir=AugConfig.TEST_MASK_DIR,
        is_train=False
    )
    
    print("\n" + "=" * 80)
    print("📊 数据增强全部完成！")
    print(f"📁 训练集图片/Mask目录：{AugConfig.TRAIN_IMG_DIR} | {AugConfig.TRAIN_MASK_DIR}")
    print(f"📁 测试集图片/Mask目录：{AugConfig.TEST_IMG_DIR} | {AugConfig.TEST_MASK_DIR}")
    print("💡 增强后文件命名规则：原名称_序号_增强类型_因子.扩展名 (例如：img_1_bright_0.5.png)")
    print("=" * 80)

if __name__ == "__main__":
    main()