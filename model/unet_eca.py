import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import time
from datetime import datetime
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random


# ===== 新增：轻量化ECA注意力模块（通道注意力，无全连接层，极轻量）=====
class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        # 自适应计算1D卷积核大小，无需手动指定
        t = int(abs((np.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        # 全局平均池化 + 1D卷积 + Sigmoid激活（核心结构，无全连接层）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        y = self.sigmoid(y)
        return x * y  # 注意力加权特征图


# ==================== UNet 模型定义 ====================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # ===== 新增：嵌入ECA注意力模块 =====
        self.eca = ECA(out_channels)

    def forward(self, x):
        x = self.double_conv(x)
        x = self.eca(x)  # 卷积后执行注意力加权
        return x

class UNet_ECA(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)
        self.pool = nn.MaxPool2d(2, 2)
        # ===== 新增：解码层专属ECA注意力（可选，轻量无负担）=====
        self.eca_dec4 = ECA(512)
        self.eca_dec3 = ECA(256)
        self.eca_dec2 = ECA(128)
        self.eca_dec1 = ECA(64)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        
        # 解码层：拼接后 + ECA注意力 + 双卷积
        d4 = self.eca_dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d4 = self.dec4(d4)
        
        d3 = self.eca_dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d3 = self.dec3(d3)
        
        d2 = self.eca_dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d2 = self.dec2(d2)
        
        d1 = self.eca_dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        d1 = self.dec1(d1)
        
        return self.out_conv(d1)