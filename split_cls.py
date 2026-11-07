from __future__ import annotations

import random
import shutil
from pathlib import Path

SRC_ROOT = Path(r"G:\UNet\data\ucf24_mini")
TEST_RATIO = 0.2
SEED = 66
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {"train", "test"}


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def main():
    classes = sorted(
        p for p in SRC_ROOT.iterdir()
        if p.is_dir() and p.name not in SPLIT_NAMES
    )
    if not classes:
        raise RuntimeError(f"{SRC_ROOT} 里没有待划分的类别文件夹")

    train_root = SRC_ROOT / "train"
    test_root = SRC_ROOT / "test"
    for folder in (train_root, test_root):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    rng = random.Random(SEED)
    for cls_dir in classes:
        images = list_images(cls_dir)
        if len(images) < 2:
            raise RuntimeError(f"{cls_dir.name} 只有 {len(images)} 张图片，无法划分")

        order = images[:]
        rng.shuffle(order)
        test_size = max(1, int(len(order) * TEST_RATIO))
        test_images = order[:test_size]
        train_images = order[test_size:]

        for split_root, chosen in ((train_root, train_images), (test_root, test_images)):
            out_dir = split_root / cls_dir.name
            out_dir.mkdir()
            for src in chosen:
                shutil.copy2(src, out_dir / src.name)

        print(f"{cls_dir.name}: train={len(train_images)} test={len(test_images)}")

    for cls_dir in classes:
        shutil.rmtree(cls_dir)

    print(f"完成: train -> {train_root}")
    print(f"完成: test  -> {test_root}")


if __name__ == "__main__":
    main()
