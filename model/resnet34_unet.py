# ==================== ResNet34-UNet 模型定义 ====================
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet34_Weights

# 残差块（ResNet34基础）
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet34-UNet主模型
class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super(ResNet34_UNet, self).__init__()
        # 加载预训练ResNet34（ImageNet权重，提升收敛）
        resnet34 = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        # 编码器：提取ResNet34的前4层特征（对应下采样4次，通道数64/128/256/512）
        self.encoder1 = nn.Sequential(
            resnet34.conv1,
            resnet34.bn1,
            resnet34.relu
        )  # (B,3,H,W) → (B,64,H/2,W/2)
        self.encoder2 = resnet34.layer1  # (B,64,H/2,W/2) → (B,64,H/2,W/2)
        self.encoder3 = resnet34.layer2  # (B,64,H/2,W/2) → (B,128,H/4,W/4)
        self.encoder4 = resnet34.layer3  # (B,128,H/4,W/4) → (B,256,H/8,W/8)
        self.encoder5 = resnet34.layer4  # (B,256,H/8,W/8) → (B,512,H/16,W/16)
        self.maxpool = resnet34.maxpool  # 配合encoder2的下采样

        # 解码器：转置卷积上采样 + 跳跃连接（UNet核心）
        # 上采样512→256，融合encoder4的256维特征
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # 上采样256→128，融合encoder3的128维特征
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # 上采样128→64，融合encoder2的64维特征
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 上采样64→64，融合encoder1的64维特征
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 输出层：64→1（二分类，对应你的water/背景分割）
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器前向，提取各层特征（保留跳跃连接用）
        e1 = self.encoder1(x)  # 64, H/2, W/2
        e2 = self.encoder2(self.maxpool(e1))  # 64, H/4, W/4
        e3 = self.encoder3(e2)  # 128, H/8, W/8
        e4 = self.encoder4(e3)  # 256, H/16, W/16
        e5 = self.encoder5(e4)  # 512, H/32, W/32

        # 解码器前向，上采样+融合跳跃特征
        d4 = self.upconv4(e5)  # 256, H/16, W/16
        d4 = torch.cat([d4, e4], dim=1)  # 256+256=512
        d4 = self.dec4(d4)  # 256, H/16, W/16

        d3 = self.upconv3(d4)  # 128, H/8, W/8
        d3 = torch.cat([d3, e3], dim=1)  # 128+128=256
        d3 = self.dec3(d3)  # 128, H/8, W/8

        d2 = self.upconv2(d3)  # 64, H/4, W/4
        d2 = torch.cat([d2, e2], dim=1)  # 64+64=128
        d2 = self.dec2(d2)  # 64, H/4, W/4

        d1 = self.upconv1(d2)  # 64, H/2, W/2
        d1 = torch.cat([d1, e1], dim=1)  # 64+64=128
        d1 = self.dec1(d1)  # 64, H/2, W/2

        # 输出层，无激活（你的DiceBCE损失用BCEWithLogitsLoss，内置sigmoid）
        out = self.out_conv(d1)
        # 尺寸对齐：若输出与输入尺寸不一致，插值匹配（复用你原有逻辑，避免报错）
        if out.shape[2:] != x.shape[2:]:
            out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out