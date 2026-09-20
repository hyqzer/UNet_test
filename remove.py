import os
import random
from pathlib import Path

# ==================== 配置区（直接复用你的数据集路径，无需修改）====================
# 数据集根路径（和你的训练脚本一致）
TRAIN_IMG_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/train_images'
TRAIN_MASK_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/train_masks'
TEST_IMG_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/test_images'
TEST_MASK_DIR = '/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/test_masks'

# 随机种子（固定种子，保证删除结果可复现，可修改/注释）
RANDOM_SEED = 42
# ==================================================================================

def set_seed(seed):
    """设置随机种子，保证删除结果可复现"""
    random.seed(seed)
    print(f"✅ 设置随机种子：{seed}，删除结果可复现")

def delete_half_samples(img_dir, mask_dir, dataset_type="train"):
    """
    随机删除数据集一半的样本（图片+对应mask）
    :param img_dir: 图片目录
    :param mask_dir: mask目录
    :param dataset_type: 数据集类型（train/test），仅用于日志打印
    """
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)
    
    # 校验目录是否存在
    if not img_dir.exists() or not mask_dir.exists():
        print(f"❌ 【{dataset_type}】目录不存在，跳过：img={img_dir}, mask={mask_dir}")
        return
    
    # 获取所有图片文件（过滤隐藏文件，仅保留图片格式）
    img_suffix = {'.png', '.jpg', '.jpeg', '.bmp'}
    img_files = [f for f in os.listdir(img_dir) 
                 if not f.startswith('.') and Path(f).suffix.lower() in img_suffix]
    if not img_files:
        print(f"❌ 【{dataset_type}】图片目录无有效文件，跳过")
        return
    
    # 按文件名匹配mask（保证一一对应）
    valid_samples = []
    for img_name in img_files:
        mask_name = img_name  # 你的数据集图片和mask文件名完全一致，直接匹配
        mask_path = mask_dir / mask_name
        if mask_path.exists():
            valid_samples.append((img_name, mask_name))
        else:
            print(f"⚠️  【{dataset_type}】mask不存在，跳过该样本：{mask_name}")
    
    if not valid_samples:
        print(f"❌ 【{dataset_type}】无有效图片-mask匹配对，跳过")
        return
    
    # 统计原始样本数
    total_num = len(valid_samples)
    delete_num = total_num // 2  # 严格删除一半（向下取整）
    remain_num = total_num - delete_num
    print(f"\n===== 【{dataset_type}集】开始处理 =====")
    print(f"原始有效样本数：{total_num}")
    print(f"计划删除样本数：{delete_num}（50%）")
    print(f"计划剩余样本数：{remain_num}")
    
    # 随机筛选要删除的样本索引
    random_idx = random.sample(range(total_num), delete_num)
    delete_samples = [valid_samples[i] for i in random_idx]
    
    # 执行删除
    deleted_count = 0
    for img_name, mask_name in delete_samples:
        try:
            # 删除图片
            os.remove(img_dir / img_name)
            # 删除对应mask
            os.remove(mask_dir / mask_name)
            deleted_count += 1
        except Exception as e:
            print(f"⚠️  删除失败：{img_name}/{mask_name}，错误：{e}")
    
    # 最终校验
    final_img_num = len([f for f in os.listdir(img_dir) if not f.startswith('.')])
    final_mask_num = len([f for f in os.listdir(mask_dir) if not f.startswith('.')])
    print(f"===== 【{dataset_type}集】处理完成 =====")
    print(f"实际删除样本数：{deleted_count}")
    print(f"最终图片数：{final_img_num}")
    print(f"最终mask数：{final_mask_num}")
    print(f"✅ 图片/mask数量匹配：{final_img_num == final_mask_num}\n")

if __name__ == "__main__":
    # 固定随机种子（可选，注释后每次删除结果不同）
    set_seed(RANDOM_SEED)
    
    # 批量处理训练集+测试集
    delete_half_samples(TRAIN_IMG_DIR, TRAIN_MASK_DIR, dataset_type="训练")
    delete_half_samples(TEST_IMG_DIR, TEST_MASK_DIR, dataset_type="测试")
    
    print("🎉 所有数据集删除操作完成！")