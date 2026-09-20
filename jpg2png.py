import os
from PIL import Image, ImageOps
from pathlib import Path

def convert_jpg_to_png(
    input_dir,          # 待转换的图片文件夹路径
    output_dir=None,    # 转换后保存的文件夹（None则保存在原文件夹）
    delete_original=True  # 改为默认删除原JPG文件
):
    """
    批量将JPG/JPEG图片无损转换为PNG格式，自动校正方向（解决上下颠倒）
    转换成功后自动删除原JPG文件
    :param input_dir: 输入文件夹路径
    :param output_dir: 输出文件夹路径（None则和原文件同目录）
    :param delete_original: 是否删除原JPG文件，默认True（删除）
    """
    # 处理输出目录：默认和输入目录相同
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有JPG/JPEG文件
    jpg_extensions = (".jpg", ".jpeg")
    jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith(jpg_extensions)]
    
    if not jpg_files:
        print(f"⚠️  在 {input_dir} 中未找到JPG/JPEG格式图片！")
        return

    # 批量转换（带方向校正）
    total = len(jpg_files)
    success_count = 0
    delete_count = 0  # 新增：统计删除的原文件数
    for idx, filename in enumerate(jpg_files, 1):
        # 构建原文件路径
        jpg_path = os.path.join(input_dir, filename)
        # 生成新文件名（替换后缀为.png）
        file_stem = Path(filename).stem  # 提取文件名（不含后缀）
        png_filename = f"{file_stem}.png"
        png_path = os.path.join(output_dir, png_filename)

        try:
            # 读取JPG图片 + 自动校正EXIF方向（核心修复）
            with Image.open(jpg_path) as img:
                # 步骤1：应用EXIF方向校正（解决大部分颠倒问题）
                img = ImageOps.exif_transpose(img)
                # 步骤2：额外检测并反转（应对少数存储顺序问题）
                # 若仍有颠倒，可取消下面注释（根据实际情况选择）
                # img = img.transpose(Image.FLIP_TOP_BOTTOM)
                
                # 保存为PNG（无损）
                img.save(png_path, format="PNG", optimize=False)
            
            success_count += 1
            print(f"✅ 转换进度：{idx}/{total} | {filename} → {png_filename}")

            # 仅在转换成功后删除原JPG文件（关键：避免转换失败丢文件）
            if delete_original:
                if os.path.exists(jpg_path):  # 二次确认文件存在
                    os.remove(jpg_path)
                    delete_count += 1
                    print(f"   🗑️  已删除原JPG文件：{filename}")

        except Exception as e:
            print(f"❌ 转换失败：{filename} | 错误信息：{str(e)}")
            print(f"   ⚠️  原JPG文件未删除：{filename}")

    # 输出结果总结
    print(f"\n🎉 转换完成！")
    print(f"✅ 成功转换：{success_count}/{total} 张图片")
    print(f"🗑️  成功删除：{delete_count} 个原JPG文件")
    print(f"📂 转换后文件保存至：{output_dir}")

# -------------------------- 配置参数（无需修改，直接运行） --------------------------
if __name__ == "__main__":
    # 待转换的文件夹路径
    INPUT_DIR = r"/home/ytkj/work/hangyq/Projects/Datasets/0119/xuzhou/images"
    # 转换后保存的文件夹（None则保存在原文件夹）
    OUTPUT_DIR = None
    # 默认删除原JPG文件（已设为True，无需修改）
    DELETE_ORIGINAL = True

    # 调用转换函数
    convert_jpg_to_png(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        delete_original=DELETE_ORIGINAL
    )