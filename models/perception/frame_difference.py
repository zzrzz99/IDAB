"""
帧间差分模块（Inter-frame Difference）
计算相邻帧全局特征的差异，作为生成状态感知的输入
"""
import torch
import torch.nn as nn


class InterFrameDifference(nn.Module):
    """计算相邻帧特征差异：D_t = phi(F_{t+1} - F_t)"""

    def __init__(self, in_channels=None, use_norm=False):
        """
        Args:
            in_channels: 输入特征通道数（可选，用于 1x1 卷积调整）
            use_norm: 是否对差分结果做归一化
        """
        super().__init__()
        self.use_norm = use_norm
        self.in_channels = in_channels
        if in_channels is not None:
            self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        else:
            self.proj = None
        if use_norm:
            self.norm = nn.BatchNorm2d(in_channels) if in_channels else None

    def forward(self, Fg):
        """
        Args:
            Fg: (B, T, C, H, W) 全局特征序列
        Returns:
            delta_F: (B, T-1, C, H, W) 帧间差分特征
        """
        B, T, C, H, W = Fg.shape
        F_curr = Fg[:, :-1].contiguous().view(B * (T - 1), C, H, W)
        F_next = Fg[:, 1:].contiguous().view(B * (T - 1), C, H, W)
        delta = torch.abs(F_next - F_curr)
        if self.proj is not None:
            delta = self.proj(delta)
        if self.use_norm and self.norm is not None:
            delta = self.norm(delta)
        return delta.view(B, T - 1, C, H, W)


# 兼容性别名
FrameDifference = InterFrameDifference
FrameDiff = InterFrameDifference
