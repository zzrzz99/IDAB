"""
EfficientNet-B4 编码器（旧版 v1，已废弃）
"""
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetB4V1(nn.Module):
    def __init__(self):
        super(EfficientNetB4V1, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.feature_dim = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()  # 去掉原有分类头

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入图像
        Returns:
            l2_feat: (B, 32, H2, W2) L2 层特征
            l5_feat: (B, 160, H5, W5) L5 层特征
        """
        l2_feat = None
        l5_feat = None
        for i, block in enumerate(self.backbone._blocks):
            x = block(x)
            if i == 3:  # L2
                l2_feat = x
            if i == 16:  # L5
                l5_feat = x
        return l2_feat, l5_feat
