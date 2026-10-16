"""
生成状态感知模块（Generation State Perception Module, GSPM）
整合帧间差分与状态注意力生成，输出时空状态注意力图 A_st
"""
import torch
import torch.nn as nn
from .frame_difference import InterFrameDifference
from .state_attention import StateAttentionGenerator


class GSPM(nn.Module):
    """
    GSPM: 从全局特征序列 Fg 感知生成状态
    输入: Fg (B, T, C, H, W)
    输出: A_st (B, T, 1, H, W)
    """

    def __init__(self, in_channels, hidden=64, align_mode='repeat_first'):
        super().__init__()
        self.frame_difference = InterFrameDifference(in_channels=in_channels)
        self.state_attention = StateAttentionGenerator(
            in_channels=in_channels, hidden=hidden, align_mode=align_mode
        )

    def forward(self, Fg):
        """
        Args:
            Fg: (B, T, C, H, W) 全局特征序列
        Returns:
            A_st: (B, T, 1, H, W) 时空状态注意力图
        """
        delta_F = self.frame_difference(Fg)
        A_st = self.state_attention(delta_F, Fg)
        return A_st
