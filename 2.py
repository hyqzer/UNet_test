from __future__ import annotations

import shutil
from pathlib import Path

SRC_ROOT = Path(r"G:\database\ucf24\rgb-images")
DST_ROOT = Path(r"G:\UNet\data\ucf24_mini")
NUM_CLASSES = 10
NUM_IMAGES = 10
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def main():
    classes = sorted(p for p in SRC_ROOT.iterdir() if p.is_dir())[:NUM_CLASSES]
    if len(classes) < NUM_CLASSES:
        raise RuntimeError(f"只有 {len(classes)} 个类别，不足 {NUM_CLASSES} 个")

    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)
    DST_ROOT.mkdir(parents=True)

    for cls_dir in classes:
        videos = sorted(p for p in cls_dir.iterdir() if p.is_dir())
        out_dir = DST_ROOT / cls_dir.name
        out_dir.mkdir()
        picked = 0

        for video_dir in videos:
            if picked >= NUM_IMAGES:
                break
            images = list_images(video_dir)
            if not images:
                continue
            src_img = images[0]
            dst_img = out_dir / f"{video_dir.name}_{src_img.name}"
            shutil.copy2(src_img, dst_img)
            picked += 1

        print(f"{cls_dir.name}: {picked}")
        if picked < NUM_IMAGES:
            raise RuntimeError(f"{cls_dir.name} 只找到 {picked} 张图片")

    print(f"完成: {NUM_CLASSES} 类 x {NUM_IMAGES} 张 -> {DST_ROOT}")


if __name__ == "__main__":
    main()
