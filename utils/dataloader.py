import os
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image


import os
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class Datasets(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir/
            cat/
                xxx.jpg
                yyy.jpg
            dog/
                ...
        """
        self.root_dir = root_dir
        self.transform = transform

        # 每个子文件夹名 = 一个类别
        self.class_names = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # 遍历所有图片，收集 (path, label)
        self.sample = []
        for cls_name in self.class_names:
            cls_path = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                ext = os.path.splitext(img_name)[1].lower()
                if os.path.isfile(img_path) and ext in IMG_EXTS:
                    self.sample.append([img_path, self.class_to_idx[cls_name]])

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        img_path, label = self.sample[idx]          # ← 方括号
        image = Image.open(img_path).convert("RGB")  # HWC

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

class DatasLoader(DataLoader):
    pass