import os

import torch

from torchvision import transforms
from torch.utils.data import DataLoader

from utils.dataloader import Datasets
from model.unet_cls import apply_lora, build_unet_classifier, set_backbone_frozen


def load_pretrained(model, path, model_type, class_names):
    """加载分类权重。类别不一致时跳过 fc，只恢复 backbone。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"无法读取分类权重: {path}")

    ckpt_type = ckpt.get("model_type")
    if ckpt_type is not None and ckpt_type != model_type:
        raise ValueError(f"权重 model_type={ckpt_type}，当前模型是 {model_type}")

    state = ckpt["model"]
    current = model.state_dict()
    ckpt_classes = ckpt.get("class_names")
    same_classes = ckpt_classes == list(class_names)

    filtered = {}
    skipped = []
    for key, value in state.items():
        if key not in current or current[key].shape != value.shape:
            skipped.append(key)
            continue
        if key.startswith("fc.") and not same_classes:
            skipped.append(key)
            continue
        filtered[key] = value

    model.load_state_dict(filtered, strict=False)
    missing_backbone = [
        key for key in current
        if key.startswith("encoder.")
        and "lora_A" not in key
        and "lora_B" not in key
        and key not in filtered
    ]
    if missing_backbone:
        raise RuntimeError(
            f"backbone 没有完整加载，缺 {len(missing_backbone)} 个参数，例如 {missing_backbone[:3]}"
        )
    return ckpt, skipped


def train(img_train, img_val, save_path, model_type="n", epochs=10, batch_size=12,
          pretrained=None, freeze_backbone=False, use_lora=False, lora_r=4, lora_alpha=4):
    """
    训练 RGB 分类。目录结构：

        img_train/
            cat/a.jpg
            dog/b.jpg
        img_val/
            cat/c.jpg
            dog/d.jpg

    类别名来自子文件夹，两边的文件夹名必须一致。

    freeze_backbone=True 时必须给 pretrained。此时冻住 encoder 原权重，只训练分类头。
    use_lora=True 时在冻住的卷积上加低秩支路，和分类头一起训练。
    """
    if not img_train or not img_val:
        raise ValueError("img_train 和 img_val 都要填图片根目录")
    if not save_path:
        raise ValueError("save_path 要填权重保存路径")
    if use_lora and not freeze_backbone:
        raise ValueError("use_lora=True 时要同时 freeze_backbone=True，原卷积核保持冻结")
    if use_lora and lora_r < 1:
        raise ValueError("lora_r 至少为 1")
    if freeze_backbone and not pretrained:
        raise ValueError("freeze_backbone=True 时必须传入 pretrained，指向已经训好的权重")
    if pretrained and not os.path.isfile(pretrained):
        raise FileNotFoundError(f"找不到预训练权重: {pretrained}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_data = Datasets(img_train, transform=train_transform)
    val_data = Datasets(img_val, transform=val_transform)
    if len(train_data) == 0 or len(val_data) == 0:
        raise RuntimeError("训练集或验证集没有读到图片，请检查目录和图片后缀")
    if train_data.class_names != val_data.class_names:
        raise RuntimeError(
            f"训练和验证的类别不一致: train={train_data.class_names} val={val_data.class_names}"
        )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))
    print(f"classes={train_data.class_names}")
    print(f"batch images={tuple(images.shape)} {images.dtype} | labels={tuple(labels.shape)} {labels.dtype}")

    model = build_unet_classifier(
        model_type=model_type,
        in_ch=3,
        class_num=len(train_data.class_names),
    ).to(device)
    if use_lora:
        apply_lora(model.encoder, r=lora_r, alpha=lora_alpha)
        model.to(device)

    if pretrained:
        ckpt, skipped = load_pretrained(model, pretrained, model_type, train_data.class_names)
        print(
            f"loaded {pretrained} | epoch={ckpt.get('epoch')} "
            f"best_acc={ckpt.get('best_acc')} classes={ckpt.get('class_names')}"
        )
        if skipped:
            print(f"skipped {len(skipped)} keys: {skipped}")

    set_backbone_frozen(model, freeze_backbone)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("没有可训练参数")

    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_count = sum(p.numel() for p in trainable) / 1e6
    print(
        f"model type = {model_type}, params = {total:.2f} M, "
        f"trainable = {trainable_count:.4f} M, freeze_backbone = {freeze_backbone}, "
        f"use_lora = {use_lora}, device = {device}"
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        if freeze_backbone:
            model.encoder.eval()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for image, label in train_loader:
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).long().view(-1)

            optimizer.zero_grad()
            logits = model(image)   # [batch, num_classes]  前向传播
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            tr_loss += loss.item() * image.size(0)
            tr_correct += (logits.argmax(1) == label).sum().item()
            tr_total += label.size(0)

        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total

        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0

        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True).long().view(-1)

                logits = model(image)
                loss = criterion(logits, label)

                va_loss += loss.item() * image.size(0)
                va_correct += (logits.argmax(1) == label).sum().item()
                va_total += label.size(0)

        va_loss /= va_total
        va_acc = va_correct / va_total
        scheduler.step()

        print(
            f"[{epoch + 1:>3}/{epochs}] "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
            f"va_loss={va_loss:.4f} va_acc={va_acc:.4f}"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "best_acc": best_acc,
                "class_names": train_data.class_names,
                "model_type": model_type,
                "freeze_backbone": freeze_backbone,
                "pretrained": pretrained,
                "use_lora": use_lora,
                "lora_r": lora_r if use_lora else None,
                "lora_alpha": lora_alpha if use_lora else None,
            }, save_path)
            print(f"    saved {save_path}  best_acc={best_acc:.4f}")

    print(f"\n训练完成 | best val acc = {best_acc:.4f}")
    return save_path


if __name__ == "__main__":
    train(
        img_train=r"",
        img_val=r"",
        save_path=r"",
    )
