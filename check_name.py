import os
from pathlib import Path

def check_dataset_alignment(image_dir, mask_dir):
    # 获取所有文件的路径
    image_files = list(Path(image_dir).glob('*'))
    mask_files = list(Path(mask_dir).glob('*'))

    # 建立 {文件名: 后缀} 的字典
    img_dict = {f.stem: f.suffix for f in image_files if f.is_file()}
    msk_dict = {f.stem: f.suffix for f in mask_files if f.is_file()}

    img_names = set(img_dict.keys())
    msk_names = set(msk_dict.keys())

    print(f"📊 统计结果:")
    print(f"图像数量: {len(img_names)} | 掩码数量: {len(msk_names)}\n")

    error_found = False

    # 1. 检查文件名是否一一对应 (找出缺失文件)
    only_in_img = img_names - msk_names
    only_in_msk = msk_names - img_names

    if only_in_img:
        error_found = True
        print(f"❌ 以下图像缺少对应的 Mask:")
        for name in sorted(only_in_img):
            print(f"   - {name}{img_dict[name]}")

    if only_in_msk:
        error_found = True
        print(f"❌ 以下 Mask 缺少对应的图像:")
        for name in sorted(only_in_msk):
            print(f"   - {name}{msk_dict[name]}")

    # 2. 检查共同文件的类型是否匹配
    common_names = img_names & msk_names
    print(f"🔎 正在对比 {len(common_names)} 组共同文件的后缀...")
    
    mismatched_types = []
    for name in sorted(common_names):
        if img_dict[name].lower() != msk_dict[name].lower():
            mismatched_types.append((name, img_dict[name], msk_dict[name]))

    if mismatched_types:
        error_found = True
        print(f"⚠️ 类型不匹配警告 (Image vs Mask):")
        for name, img_ext, msk_ext in mismatched_types:
            print(f"   - 文件: {name} | 类型: {img_ext} vs {msk_ext}")
    
    if not error_found:
        print("✅ 完美！所有图片名称和类型均一一对应。")
    else:
        print("\n结论: 数据集存在不一致，请根据上方列表进行清理。")

# --- 使用方法 ---
if __name__ == "__main__":
    # 替换为你自己的路径
    IMGS_PATH = r"/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/images"
    MASKS_PATH = r"/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/masks"
    
    if os.path.exists(IMGS_PATH) and os.path.exists(MASKS_PATH):
        check_dataset_alignment(IMGS_PATH, MASKS_PATH)
    else:
        print("❌ 路径不存在，请检查路径配置。")