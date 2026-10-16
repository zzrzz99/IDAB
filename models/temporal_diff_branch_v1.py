"""
帧间注意力重权模块（Inter-frame Attention Reweighting，旧版 v1，已废弃）
使用 L2 特征的邻帧残差计算注意力权重，对全局特征进行重加权
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameAttentionV1(nn.Module):
    """帧间注意力模块（旧版 v1）：利用相邻帧 L2 特征差计算注意力权重"""
    def __init__(self, in_channels=32, reduction=4):
        """
        Args:
            in_channels: L2 特征通道数（默认 32）
            reduction: 通道缩减比例
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, F_curr, F_next, G_curr):
        """
        计算帧间注意力权重并重加权全局特征

        Args:
            F_curr: (B, C, H, W) 当前帧的 L2 特征
            F_next: (B, C, H, W) 下一帧的 L2 特征
            G_curr: (B, D) 当前帧的全局特征（已 GAP）
        Returns:
            F_weighted: (B, D) 重加权后的全局特征
            att: (B, 1, H, W) 注意力权重图
        """
        diff = F_next - F_curr  # (B, C, H, W)
        att = self.conv(diff)  # (B, 1, H, W)
        att_scalar = F.adaptive_avg_pool2d(att, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 1)
        F_weighted = G_curr * att_scalar  # (B, D)
        return F_weighted, att


# 保留原 TemporalDiffBranch 以保持兼容性（旧版 v1）
class TemporalDiffBranchV1(nn.Module):
    """保留原帧间差分流模块（已废弃，保留用于兼容）"""
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels*2, 3, 2, 1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*4, 3, 2, 1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True)
        )

    def forward(self, frame1, frame2):
        diff = torch.abs(frame2 - frame1)
        return self.conv(diff)
