"""
GS-Former 编码器（GSFormerEncoder）
基于 Transformer 的时序状态编码器
"""
import torch
import torch.nn as nn


class GSFormerEncoder(nn.Module):
    def __init__(self, embed_size, num_heads, num_classes, num_layers=2, dropout=0.1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_size)
        Returns:
            output: (batch, num_classes)
        """
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = self.transformer_encoder(x)
        x = self.norm(x.mean(dim=1))
        x = self.dropout(x)
        return self.fc(x)


# 兼容性别名
TransformerModel = GSFormerEncoder
