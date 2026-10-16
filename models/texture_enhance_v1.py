"""
纹理增强模块（旧版 v1，已废弃）
"""
import torch
import torch.nn as nn


class TextureEnhanceBlockV1(nn.Module):
    """纹理增强块（旧版 v1）"""
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels*2, in_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels*3, in_channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征
        Returns:
            out: (B, C, H, W) 增强后的纹理特征
        """
        D = self.pool(x)
        texture = x - D
        out1 = self.relu(self.conv1(texture))
        out2 = self.relu(self.conv2(torch.cat([texture, out1], dim=1)))
        out3 = self.relu(self.conv3(torch.cat([texture, out1, out2], dim=1)))
        return out3
