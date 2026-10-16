"""
ML-FFormer 模块（旧版 v1，已废弃）
多标签分类的 Transformer 编码器
"""
import torch
import torch.nn as nn
from .transformer_v1 import TransformerModelV1


class MLFFormerV1(nn.Module):
    """多标签分类的 Transformer 编码器（旧版 v1）"""
    def __init__(self, embed_size=256, num_heads=8, num_classes=5, num_layers=2, dropout: float = 0.1):
        super().__init__()
        self.transformer = TransformerModelV1(
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
            tokens = tokens.unsqueeze(1)  # (B, 1, embed_size)
        logits = self.transformer(tokens)
        return logits
