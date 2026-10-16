"""
生成状态增强模块（Generation State Enhancement Module, GSEM）
使用状态注意力图增强纹理特征：T̃_t = A_t ⊙ T_t
"""
import torch
import torch.nn as nn


class GSEM(nn.Module):
    """状态引导的纹理特征增强"""

    def __init__(self, lambda_init=1.0, learnable=True):
        """
        Args:
            lambda_init: 增强强度初始值
            learnable: lambda 是否可学习
        """
        super().__init__()
        if learnable:
            self.lambda_ = nn.Parameter(torch.tensor(float(lambda_init)))
        else:
            self.register_buffer('lambda_', torch.tensor(float(lambda_init)))

    def forward(self, Ft, A_st):
        """
        Args:
            Ft: (B, T, C, H, W) 纹理特征
            A_st: (B, T, 1, H, W) 时空状态注意力图
        Returns:
            F_tex_enh: (B, T, C, H, W) 增强后的纹理特征
        """
        scale = 1.0 + self.lambda_ * torch.sigmoid(A_st)
        return Ft * scale


# 兼容性别名
TextureEnhancement = GSEM
AttentionGuidedTextureEnhance = GSEM
