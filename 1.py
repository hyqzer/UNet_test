import os

# 你的目标文件夹路径
target_dir = "/home/ytkj/work/hangyq/Projects/Datasets/0119/20260228/images"

# 遍历指定目录
for file in os.listdir(target_dir):
    # 判断是否以 .json 结尾
    if file.endswith('.jpg'):
        # 拼接完整文件路径
        file_path = os.path.join(target_dir, file)
        try:
            os.remove(file_path)
            print(f'已删除: {file_path}')
        except Exception as e:
            print(f'删除失败 {file_path}: {e}')

print('操作完成')