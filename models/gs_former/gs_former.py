"""
GS-Former（Generation State Former）
循环状态更新 + 状态提示交叉注意力 + Transformer 编码器
"""
import torch
import torch.nn as nn
from .gs_former_encoder import GSFormerEncoder


class GSFormer(nn.Module):
    """生成状态 Transformer"""

    def __init__(self, embed_size=256, num_heads=8, num_classes=5, num_layers=2, dropout=0.1):
        super().__init__()
        self.transformer = GSFormerEncoder(
            embed_size=embed_size,
            num_heads=num_heads,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, tokens):
        """
        Args:
            tokens: (B, num_tokens, embed_size) 输入 tokens
        Returns:
            logits: (B, num_classes) 未经过 Sigmoid 的 logits
        """
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(1)
        logits = self.transformer(tokens)
        return logits


# 兼容性别名
MLFFormer = GSFormer
