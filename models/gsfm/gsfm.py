"""
生成状态融合模块（Generation State Fusion Module, GSFM）
通过双线性外积融合纹理与全局特征，并用状态注意力池化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GSFM(nn.Module):
    """注意力引导的双线性池化融合"""

    def __init__(self, C_tex, C_glob, d=32):
        """
        Args:
            C_tex: 纹理特征通道数
            C_glob: 全局特征通道数
            d: 投影维度，输出 Z [B, T, d*d]
        """
        super().__init__()
        self.d = d
        self.proj_tex = nn.Conv2d(C_tex, d, kernel_size=1)
        self.proj_glob = nn.Conv2d(C_glob, d, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, F_tex, F_glob, A_st):
        """
        Args:
            F_tex: (B, T, C_tex, H, W) 纹理特征
            F_glob: (B, T, C_glob, H, W) 全局特征
            A_st: (B, T, 1, H, W) 状态注意力图
        Returns:
            Z: (B, T, d*d) 双线性融合特征
        """
        B, T, _, H, W = F_tex.shape
        Ft = self.proj_tex(F_tex.view(B * T, -1, H, W))
        Fg = self.proj_glob(F_glob.view(B * T, -1, H, W))
        # 外积 (B*T, d, H*W) x (B*T, H*W, d) -> (B*T, d, d)
        Ft_flat = Ft.view(B * T, self.d, -1)
        Fg_flat = Fg.view(B * T, self.d, -1)
        B_t = torch.bmm(Ft_flat, Fg_flat.transpose(1, 2))
        # 状态注意力加权池化
        att = A_st.view(B * T, 1, H * W)
        att = att / (att.sum(dim=-1, keepdim=True) + 1e-6)
        pooled = torch.bmm(B_t.view(B * T, self.d * self.d, 1), att).squeeze(-1)
        # signed sqrt + L2 norm
        Z = torch.sign(pooled) * torch.sqrt(torch.abs(pooled) + 1e-6)
        Z = F.normalize(Z, p=2, dim=-1)
        return Z.view(B, T, self.d * self.d)


# 兼容性别名
AttentionGuidedBilinearPooling = GSFM
