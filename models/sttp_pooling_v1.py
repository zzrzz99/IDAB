"""
时空 Transformer 池化模块（旧版 v1，已废弃）
利用轻量 Transformer 在时空维上做"软池化"聚合，代替平均或 attention 池化
"""
import torch
import torch.nn as nn


class SpatioTemporalTransformerPoolingV1(nn.Module):
    def __init__(self, dim, num_heads=4, depth=2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=dim * 4,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, features):
        """
        Args:
            features: (B, T, D) 表示每帧的融合特征
        Returns:
            pooled: (B, D) 时空聚合后的特征
        """
        x = self.transformer(features)  # 时序依赖建模
        pooled = self.fc(torch.mean(x, dim=1))  # 类似"软池化"
        return pooled
