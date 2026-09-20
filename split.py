import os
import shutil
import random
from glob import glob
from natsort import natsorted  # 保证文件名有序拆分

# ====================== 配置参数（根据你的需求修改）======================
# 根路径（你的20260228文件夹路径）
ROOT_DIR = "/home/ytkj/work/hangyq/Projects/Datasets/0119/qiangwei"
# 测试集比例（比如拆分20%作为测试集，80%作为训练集）
TEST_RATIO = 0.2
# 图片格式（根据你的实际格式修改，比如.png/.jpg）
IMG_EXT = ".png"
# 是否打乱数据（建议True，保证拆分随机性）
SHUFFLE = True
# 随机种子（固定种子保证拆分结果可复现）
SEED = 66

# ====================== 自动创建目标目录 ======================

TARGET_DIRS = {
    "train_images": os.path.join(ROOT_DIR, "train_images"),
    "train_masks": os.path.join(ROOT_DIR, "train_masks"),
    "test_images": os.path.join(ROOT_DIR, "test_images"),
    "test_masks": os.path.join(ROOT_DIR, "test_masks"),
}

# 创建目录（如果已存在则清空，避免重复）
for dir_path in TARGET_DIRS.values():
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    print(f"创建目录: {dir_path}")

# ====================== 读取并校验原始数据 ======================
# 读取所有图片和标签（按名称排序，保证一一对应）
img_paths = natsorted(glob(os.path.join(ROOT_DIR, "images", f"*{IMG_EXT}")))
mask_paths = natsorted(glob(os.path.join(ROOT_DIR, "masks", f"*{IMG_EXT}")))

# 校验：图片和标签数量必须一致
assert len(img_paths) == len(mask_paths), \
    f"图片数量({len(img_paths)})和标签数量({len(mask_paths)})不匹配！"
assert len(img_paths) > 0, "images文件夹中没有找到图片！"

# 提取文件名（用于校验图片和标签名称一致）
img_names = [os.path.basename(p) for p in img_paths]
mask_names = [os.path.basename(p) for p in mask_paths]
assert img_names == mask_names, "图片和标签的文件名不一一对应！"

# ====================== 拆分训练集/测试集 ======================
# 固定随机种子
random.seed(SEED)
# 生成索引并打乱（如果需要）
indices = list(range(len(img_paths)))
if SHUFFLE:
    random.shuffle(indices)

# 计算测试集数量
test_size = int(len(indices) * TEST_RATIO)
# 拆分索引（前test_size个为测试集，剩余为训练集）
test_indices = indices[:test_size]
train_indices = indices[test_size:]

print(f"\n数据集拆分完成：")
print(f"总数据量：{len(img_paths)}")
print(f"训练集数量：{len(train_indices)}")
print(f"测试集数量：{len(test_indices)}")

# ====================== 复制文件到目标目录 ======================
def copy_files(indices, src_img_paths, src_mask_paths, dst_img_dir, dst_mask_dir):
    """批量复制图片和标签到目标目录"""
    for idx in indices:
        # 图片
        src_img = src_img_paths[idx]
        dst_img = os.path.join(dst_img_dir, os.path.basename(src_img))
        shutil.copyfile(src_img, dst_img)
        # 标签
        src_mask = src_mask_paths[idx]
        dst_mask = os.path.join(dst_mask_dir, os.path.basename(src_mask))
        shutil.copyfile(src_mask, dst_mask)

# 复制训练集
print("\n开始复制训练集...")
copy_files(
    train_indices,
    img_paths, mask_paths,
    TARGET_DIRS["train_images"], TARGET_DIRS["train_masks"]
)

# 复制测试集
print("开始复制测试集...")
copy_files(
    test_indices,
    img_paths, mask_paths,
    TARGET_DIRS["test_images"], TARGET_DIRS["test_masks"]
)

# ====================== 最终校验 ======================
train_img_count = len(os.listdir(TARGET_DIRS["train_images"]))
train_mask_count = len(os.listdir(TARGET_DIRS["train_masks"]))
test_img_count = len(os.listdir(TARGET_DIRS["test_images"]))
test_mask_count = len(os.listdir(TARGET_DIRS["test_masks"]))

assert train_img_count == train_mask_count == len(train_indices), "训练集复制失败！"
assert test_img_count == test_mask_count == len(test_indices), "测试集复制失败！"

print("\n✅ 数据集拆分完成！最终目录结构：")
print(f"└── 20260228")
print(f"    ├── train_images ({train_img_count}张)")
print(f"    ├── train_masks ({train_mask_count}张)")
print(f"    ├── test_images ({test_img_count}张)")
print(f"    ├── test_masks ({test_mask_count}张)")
print(f"    ├── test_image (原有测试图)")
print(f"    ├── test_label (原有测试标签)")
print(f"    ├── images (原始图片)")
print(f"    └── masks (原始标签)")