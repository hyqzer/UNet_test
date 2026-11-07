from __future__ import annotations

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# ----------------- 分割函数 ---------------
"""
输入 (3, 224, 224)
  │
  ├─ Encoder（下采样）
  │   conv → conv → pool        (64, 112, 112)
  │   conv → conv → pool        (128, 56, 56)
  │   conv → conv → pool        (256, 28, 28)
  │   conv → conv → pool        (512, 14, 14)
  │   conv → conv               (1024, 7, 7)   ← bottleneck
  │
  └─ Decoder（上采样 + 跳跃连接）
      up + concat(512) → conv → conv
      up + concat(256) → conv → conv
      up + concat(128) → conv → conv
      up + concat(64)  → conv → conv
      → 1x1 conv → 输出 mask
"""


# ----------------- 分类函数 ---------------
"""
输入 (3, 224, 224)
  │
  ├─ Encoder（下采样）
  │   conv → conv → pool        (64, 112, 112)  C H W  这是输出
  │   conv → conv → pool        (128, 56, 56)
  │   conv → conv → pool        (256, 28, 28)
  │   conv → conv → pool        (512, 14, 14)
  │   conv → conv               (1024, 7, 7)   ← bottleneck
  │
  └─ GAP (Global Average Pooling) → Linear → 类别数
"""


class BottlenDoubleConv(nn.Module):
    pass


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, depth=4):
        super().__init__()

        self.enc = nn.ModuleList()
        cur_in = in_ch
        for i in range(depth):
            self.enc.append(DoubleConv(cur_in, base_ch*(2**i)))
            cur_in = base_ch*(2**i)

        
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(cur_in, base_ch*(2**depth))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        output_list = []

        for blocks in self.enc:
            x = blocks(x)
            output_list.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        output_list.append(x)

        return output_list


class UNetClassifier(nn.Module):
    def __init__(self, in_ch, num_classes=10, base_ch=64, depth=4):
        super().__init__()
        self.encoder = Encoder(in_ch, base_ch, depth)
        bot_ch = base_ch*(2**depth)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(bot_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.encoder(x)
        x2 = x1[-1]
        x3 = self.gap(x2)
        x4 = torch.flatten(x3, 1)   # [batch, feature_dim] 从第 feature_dim 维开始，把后面所有维度全部压平合并
        x5 = self.dropout(x4)
        x6 = self.fc(x5)

        return x6



class Decoder:

    pass



# ============= 工厂函数：动态构造 通道 的深度和宽度，控制 宽高 =======================================

def build_unet_classifier(model_type: str, in_ch=3, class_num=10):
    cfg={
        "n": {"base_ch":16, "depth":3},
        "s": {"base_ch":32, "depth":4},
        "m": {"base_ch":64, "depth":4},
        "l": {"base_ch":64, "depth":5},
        "x": {"base_ch":96, "depth":5},
    }

    config = cfg[model_type.lower()]

    return UNetClassifier(in_ch=in_ch, num_classes=class_num, base_ch=config["base_ch"], depth=config["depth"])


class LoRAConv2d(nn.Conv2d):
    """原卷积核保持不变，旁边加一条低秩支路：y = Wx + scale * B(A(x))。"""

    def __init__(self, *args, r=4, alpha=4, **kwargs):
        super().__init__(*args, **kwargs)
        if r < 1:
            raise ValueError("lora r 至少为 1")
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Conv2d(
            self.in_channels,
            r,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=False,
        )
        self.lora_B = nn.Conv2d(r, self.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, r: int, alpha: int) -> "LoRAConv2d":
        lora = cls(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            r=r,
            alpha=alpha,
        )
        lora.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            lora.bias.data.copy_(conv.bias.data)
        return lora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + self.scale * self.lora_B(self.lora_A(x))


def apply_lora(module: nn.Module, r=4, alpha=4) -> None:
    """把模块里的 Conv2d 换成 LoRAConv2d。已经换过的层不会再套一层。"""
    for name, child in list(module.named_children()):
        if isinstance(child, LoRAConv2d):
            continue
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d.from_conv(child, r, alpha))
        else:
            apply_lora(child, r, alpha)


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    """冻住 encoder 的原权重。LoRA 支路和分类头保持可训练。"""
    for name, p in model.encoder.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
            continue
        p.requires_grad = not frozen


