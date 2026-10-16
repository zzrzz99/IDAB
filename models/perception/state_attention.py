"""
生成状态注意力模块（State Attention）
从帧间差分特征生成注意力权重，用于增强和融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateWeightGenerator(nn.Module):
    """从帧间差分生成状态权重：alpha = f(delta_F)"""

    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, delta_F):
        """
        Args:
            delta_F: (B, T-1, C, H, W) 帧间差分特征
        Returns:
            alpha: (B, T-1, 1, H, W) 状态注意力权重
        """
        B, Tm1, C, H, W = delta_F.shape
        x = delta_F.view(B * Tm1, C, H, W)
        alpha = self.conv(x)
        return alpha.view(B, Tm1, 1, H, W)


class StateWeightAlignment(nn.Module):
    """将状态权重对齐到帧级（首帧复制或插值）"""

    def __init__(self, mode='repeat_first'):
        super().__init__()
        self.mode = mode

    def forward(self, alpha):
        """
        Args:
            alpha: (B, T-1, 1, H, W) 帧间注意力
        Returns:
            alpha_frame: (B, T, 1, H, W) 帧级注意力
        """
        if self.mode == 'repeat_first':
            first = alpha[:, :1]
            return torch.cat([first, alpha], dim=1)
        elif self.mode == 'pad_zero':
            B, Tm1, _, H, W = alpha.shape
            pad = torch.zeros(B, 1, 1, H, W, device=alpha.device, dtype=alpha.dtype)
            return torch.cat([pad, alpha], dim=1)
        else:
            raise ValueError(f"Unknown alignment mode: {self.mode}")


class StateAttentionPooling(nn.Module):
    """使用状态注意力对全局特征做加权池化"""

    def __init__(self):
        super().__init__()

    def forward(self, Fg, alpha):
        """
        Args:
            Fg: (B, T, C, H, W) 全局特征
            alpha: (B, T, 1, H, W) 帧级状态注意力
        Returns:
            A: (B, T, C) 注意力池化后的特征
        """
        weighted = Fg * alpha
        return weighted.mean(dim=(-2, -1))


class StateAttentionGenerator(nn.Module):
    """整合：delta_F -> alpha -> alpha_frame -> A_st"""

    def __init__(self, in_channels, hidden=64, align_mode='repeat_first'):
        super().__init__()
        self.weight_gen = StateWeightGenerator(in_channels, hidden)
        self.alignment = StateWeightAlignment(mode=align_mode)
        self.pooling = StateAttentionPooling()

    def forward(self, delta_F, Fg):
        """
        Args:
            delta_F: (B, T-1, C, H, W) 帧间差分
            Fg: (B, T, C, H, W) 全局特征
        Returns:
            A_st: (B, T, 1, H, W) 时空状态注意力图
        """
        alpha = self.weight_gen(delta_F)
        alpha_frame = self.alignment(alpha)
        return alpha_frame


# 兼容性别名
TemporalWeightGenerator = StateWeightGenerator
TemporalAlignment = StateWeightAlignment
AttentionPooling = StateAttentionPooling
TemporalAttentionGenerator = StateAttentionGenerator
