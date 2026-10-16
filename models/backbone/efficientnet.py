"""
Backbone：单帧特征提取
只用一个 CNN，在中间 block（L2/L3）截取纹理特征，L5 为全局特征。
输入 [B, T, 3, H, W] 经单次前向得到 Fg、Ft。
"""
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetBackbone(nn.Module):
    """
    单 CNN 输出：
    - Global feature Fg: [B, T, Cg, Hg, Wg]  (L5, Cg=160)
    - Texture feature Ft: [B, T, Ct, Ht, Wt] (L2, Ct=32)
    """

    def __init__(self, texture_block_idx=3, global_block_idx=16):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.backbone._fc = nn.Identity()
        self.texture_block_idx = texture_block_idx
        self.global_block_idx = global_block_idx
        self.Cg = 160
        self.Ct = 32

    def forward(self, x):
        """
        Args:
            x: (B, T, 3, H, W) 或 (B*T, 3, H, W)
        Returns:
            Fg: (B, T, Cg, Hg, Wg)
            Ft: (B, T, Ct, Ht, Wt)
        """

        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            need_reshape = True
        else:
            need_reshape = False
            B, T = x.size(0), 1

        # Stem（与 EfficientNet 官方一致：3 通道 → _blocks 输入）
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        Fg = None
        Ft = None
        for i, block in enumerate(self.backbone._blocks):
            x = block(x)
            if i == self.texture_block_idx:
                Ft = x
            if i == self.global_block_idx:
                Fg = x

        if need_reshape:
            Ft = Ft.view(B, T, *Ft.shape[1:])
            Fg = Fg.view(B, T, *Fg.shape[1:])
        return Fg, Ft
