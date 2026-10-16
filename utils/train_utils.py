"""
训练工具函数
从原始 ETMA 复制并适配（旧版术语，功能不变）
"""
import torch


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def adjust_lr(optimizer, epoch, initial_lr, lr_decay_epoch=10, decay_rate=0.1):
    lr = initial_lr * (decay_rate ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

