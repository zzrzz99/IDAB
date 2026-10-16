"""
生成状态聚合模块（Generation State Aggregation Module, GSAM）
通过 K 分支注意力将特征图聚合为 tokens
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GSAM(nn.Module):
    """K 分支注意力聚合器"""

    def __init__(self, in_dim, num_tokens=5, eps=1e-6):
        super().__init__()
        self.num_tokens = num_tokens
        self.attn_conv = nn.Conv2d(in_dim, num_tokens, kernel_size=1)
        self.eps = eps

    def forward(self, feat, attention_map=None):
        """
        Args:
            feat: (B, C, H, W) 输入特征
            attention_map: (B, 1, H, W) 可选外部引导注意力图
        Returns:
            tokens: (B, num_tokens, C) 聚合后的 tokens
        """
        B, C, H, W = feat.shape
        attn_logits = self.attn_conv(feat).view(B, self.num_tokens, -1)
        attn_map = torch.softmax(attn_logits, dim=-1)

        if attention_map is not None:
            if attention_map.dim() == 3:
                attention_map = attention_map.unsqueeze(1)
            if attention_map.shape[-2:] != (H, W):
                raise ValueError("attention_map spatial size must match feat")
            guide = attention_map.view(B, 1, -1)
            attn_map = attn_map * guide
            attn_sum = attn_map.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            attn_map = attn_map / attn_sum

        feat_flat = feat.view(B, C, -1)
        tokens = torch.bmm(attn_map, feat_flat.transpose(1, 2))
        return tokens


# 兼容性别名
MAA = GSAM
TokenAttentionAggregator = GSAM
