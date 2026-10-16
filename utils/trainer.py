"""
兼容入口，重导出 trainer_core 中的训练与验证函数。
"""
from .trainer_core import train_epoch, validate

__all__ = ['train_epoch', 'validate']

