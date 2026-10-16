"""
注意力加权融合模块（旧版 v1，已废弃）
用全局特征作为 Query，帧间特征作为 Key/Value，通过注意力实现特征选择
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusionV1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, Fg, Ft):
        """
        Args:
            Fg: (B, N, D) 全局特征（Global features）
            Ft: (B, M, D) 帧间特征（Temporal difference features）
        Returns:
            out: (B, N, D) 融合后的特征
        """
        Q = self.query(Fg)
        K = self.key(Ft)
        V = self.value(Ft)
        attn = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1)
        out = torch.bmm(attn, V)
        out = self.proj(out) + Fg  # 残差连接
        return out
