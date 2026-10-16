"""
Transformer 模型（旧版 v1，已废弃）
"""
import torch
import torch.nn as nn


class TransformerModelV1(nn.Module):
    def __init__(self, embed_size, num_heads, num_classes, num_layers=2, dropout: float = 0.1):
        super(TransformerModelV1, self).__init__()
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
        x = x.unsqueeze(1) if x.dim() == 2 else x  # (batch, 1, embed_size)
        x = self.transformer_encoder(x)
        x = self.norm(x.mean(dim=1))
        x = self.dropout(x)
        return self.fc(x)
