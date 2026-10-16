"""
数据集加载器
从原始 ETMA 复制并适配（旧版术语，功能不变）
支持两种目录结构：
1. 按标签分文件夹：root/split/LABEL/video_folder/*.jpg
2. Phase1 (FH-FF++)：root/split/video_xxxx/frames/*.jpg + video_label.json
"""
import os
import re
import json
import random
import logging
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

# Phase1 (run_phase0_or_phase1) 输出的类别顺序，与 ffpp_config 一致
FH_LABEL_NAMES = ["DF", "F2F", "FS", "FSh", "NT", "OR"]
# 不一致性检测模式：监督信号为「帧间生成机制是否切换」，OR 不作为类别
INCONSISTENCY_LABEL_NAME = "inconsistency"


class VideoDataset(Dataset):
    def __init__(self, root_dir, label_names, split='train', transform=None, frames_per_video=16,
                 use_inconsistency_labels=False, dual_task=False):
        self.root_dir = root_dir
        self.label_names = label_names  # ['label1', 'label2', ...] 或 ['inconsistency']
        self.split = split  # 'train', 'val', 'test'
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.use_inconsistency_labels = use_inconsistency_labels
        self.dual_task = dual_task  # 双头任务：主任务帧级来源分类 + 辅任务帧间不一致性
        self.samples = self._gather_samples()
        # 估计默认帧形状（用于退化填充）
        self._default_frame_shape = self._infer_default_shape()

    def _infer_default_shape(self):
        if self.transform:
            try:
                dummy = Image.new('RGB', (256, 256))
                tensor = self.transform(dummy)
                return tuple(tensor.shape)
            except Exception:
                pass
        return (3, 256, 256)

    def _gather_samples(self):
        split_dir = os.path.join(self.root_dir, self.split)
        # 检测 Phase1 (FH-FF++) 格式：split 下直接是 video_xxxx，且存在 video_label.json
        if self._is_phase1_format(split_dir):
            return self._gather_samples_phase1(split_dir)
        # 按标签分文件夹的格式
        video_dict = {}
        for idx, label in enumerate(self.label_names):
            label_dir = os.path.join(split_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for video in os.listdir(label_dir):
                if video.startswith('.'):
                    continue
                video_path = os.path.join(label_dir, video)
                if not os.path.isdir(video_path):
                    continue
                canonical_name = self._canonical_video_name(video)
                frame_dir = self._frame_dir(video_path)
                frame_files = [
                    f for f in os.listdir(frame_dir)
                    if f.lower().endswith(('jpg', 'jpeg', 'png'))
                ]
                if not frame_files:
                    logging.warning("Skipping empty video folder: %s", video_path)
                    continue
                if canonical_name in video_dict:
                    logging.debug(
                        "Duplicate video detected (%s, canonical %s); keeping %s",
                        video,
                        canonical_name,
                        video_dict[canonical_name]['original_name'],
                    )
                    continue
                video_dict[canonical_name] = {
                    'labels': [0] * len(self.label_names),
                    'path': video_path,
                    'original_name': video,
                }
                video_dict[canonical_name]['labels'][idx] = 1
        samples = []
        for _, info in video_dict.items():
            samples.append((info['path'], info['labels']))
        return samples

    @staticmethod
    def _is_phase1_format(split_dir):
        if not os.path.isdir(split_dir):
            return False
        for d in os.listdir(split_dir):
            if d.startswith('.') or not d.startswith('video_'):
                continue
            sub = os.path.join(split_dir, d)
            if not os.path.isdir(sub):
                continue
            if os.path.isfile(os.path.join(sub, 'video_label.json')):
                return True
        return False

    def _gather_samples_phase1(self, split_dir):
        samples = []
        for video in sorted(os.listdir(split_dir)):
            if video.startswith('.') or not video.startswith('video_'):
                continue
            video_path = os.path.join(split_dir, video)
            if not os.path.isdir(video_path):
                continue
            if self.dual_task:
                # 双头任务：必须有 frame_label.json，主监督为帧级来源
                frame_label_path = os.path.join(video_path, 'frame_label.json')
                if not os.path.isfile(frame_label_path):
                    continue
                label_vec = None
            elif not self.use_inconsistency_labels:
                label_path = os.path.join(video_path, 'video_label.json')
                if not os.path.isfile(label_path):
                    continue
                try:
                    with open(label_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    labels_str = data.get('labels', [])
                except (json.JSONDecodeError, KeyError):
                    continue
                label_vec = [0] * len(self.label_names)
                for name in labels_str:
                    if name in self.label_names:
                        label_vec[self.label_names.index(name)] = 1
            else:
                # 不一致性模式：需存在 frame_label.json，标签在 __getitem__ 中从帧类型计算
                frame_label_path = os.path.join(video_path, 'frame_label.json')
                if not os.path.isfile(frame_label_path):
                    continue
                label_vec = None
            frame_dir = self._frame_dir(video_path)
            if not os.path.isdir(frame_dir):
                continue
            frame_files = [
                f for f in os.listdir(frame_dir)
                if f.lower().endswith(('jpg', 'jpeg', 'png'))
            ]
            if not frame_files:
                logging.warning("Skipping empty video folder: %s", video_path)
                continue
            samples.append((video_path, label_vec))
        return samples

    @staticmethod
    def _frame_dir(video_path):
        """帧所在目录：优先 video_path/frames（Phase1），否则 video_path"""
        frames_sub = os.path.join(video_path, 'frames')
        if os.path.isdir(frames_sub):
            return frames_sub
        return video_path

    @staticmethod
    def _canonical_video_name(name: str) -> str:
        """
        归一化视频名称：
        1. 去掉类似 _01、_02 等纯数字后缀
        2. 对类别顺序（A+B 和 B+A）进行排序
        3. 对编号对 (000_003 与 003_000) 进行排序
        """
        parts = name.split('_')
        if len(parts) > 3 and parts[-1].isdigit():
            parts = parts[:-1]
        base_name = '_'.join(parts)

        match = re.match(r'(?P<cat1>[A-Za-z]+)\+(?P<cat2>[A-Za-z]+)_(?P<id1>\d+)(?:_(?P<id2>\d+))?', base_name)
        if not match:
            return base_name.upper()

        cat1 = match.group('cat1').upper()
        cat2 = match.group('cat2').upper()
        sorted_cats = sorted([cat1, cat2])

        id1 = match.group('id1')
        id2 = match.group('id2')
        if id2 is not None:
            sorted_ids = sorted([id1, id2])
            return f"{sorted_cats[0]}+{sorted_cats[1]}_{sorted_ids[0]}_{sorted_ids[1]}"
        return f"{sorted_cats[0]}+{sorted_cats[1]}_{id1}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, labels = self.samples[idx]
        video_name = os.path.basename(video_path)
        frame_dir = self._frame_dir(video_path)
        frame_files = sorted(
            [f for f in os.listdir(frame_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        )
        if len(frame_files) == 0:
            logging.debug("No frames found in %s", video_path)
        images = []
        for frame in frame_files:
            frame_path = os.path.join(frame_dir, frame)
            try:
                with Image.open(frame_path) as img:
                    img = img.convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    else:
                        img = transforms.ToTensor()(img)
                    images.append(img)
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as err:
                logging.warning("Skipping invalid frame %s: %s", frame_path, err)
                continue
        if not images:
            logging.warning("All frames invalid for %s; returning zeros.", video_path)
            C, H, W = self._default_frame_shape
            empty = torch.zeros(self.frames_per_video, C, H, W, dtype=torch.float32)
            if self.dual_task:
                return {
                    'frames': empty,
                    'frame_labels': torch.zeros(self.frames_per_video, len(self.label_names), dtype=torch.float32),
                    'inconsistency_labels': torch.zeros(max(0, self.frames_per_video - 1), dtype=torch.float32),
                    'video_name': video_name,
                }
            if self.use_inconsistency_labels:
                return empty, torch.zeros(1, dtype=torch.float32), torch.zeros(max(0, self.frames_per_video - 1), dtype=torch.float32), video_name
            return empty, torch.tensor(labels, dtype=torch.float32), video_name

        images = torch.stack(images)  # (valid_frames, C, H, W)
        if self.split == 'train' and not self.use_inconsistency_labels and not self.dual_task:
            images = self._temporal_augment(images)
        valid_frames = images.size(0)
        if valid_frames == 1 and self.frames_per_video > 1:
            images = images.repeat(self.frames_per_video, 1, 1, 1)
            sample_indices = torch.zeros(self.frames_per_video, dtype=torch.long)
        elif valid_frames != self.frames_per_video:
            indices = torch.linspace(
                0, valid_frames - 1, steps=self.frames_per_video, dtype=torch.float32
            ).round().long()
            indices = torch.clamp(indices, 0, valid_frames - 1)
            images = images[indices]
            sample_indices = indices
        else:
            sample_indices = torch.arange(valid_frames, dtype=torch.long)

        if self.dual_task:
            frame_label_path = os.path.join(video_path, 'frame_label.json')
            try:
                with open(frame_label_path, 'r', encoding='utf-8') as f:
                    frame_label_dict = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                T = images.size(0)
                return {
                    'frames': images,
                    'frame_labels': torch.zeros(T, len(self.label_names), dtype=torch.float32),
                    'inconsistency_labels': torch.zeros(max(0, T - 1), dtype=torch.float32),
                    'video_name': video_name,
                }
            frame_types = []
            for i in range(images.size(0)):
                idx = sample_indices[i].item()
                key = os.path.splitext(frame_files[idx])[0]
                if len(key) < 4:
                    key = key.zfill(4)
                frame_types.append(frame_label_dict.get(key, 'OR'))
            # frame_labels [T, C] one-hot
            C = len(self.label_names)
            frame_labels = torch.zeros(images.size(0), C, dtype=torch.float32)
            for t, ft in enumerate(frame_types):
                if ft in self.label_names:
                    frame_labels[t, self.label_names.index(ft)] = 1.0
                else:
                    frame_labels[t, self.label_names.index('OR')] = 1.0
            inconsistency = []
            for t in range(len(frame_types) - 1):
                inconsistency.append(0.0 if frame_types[t] == frame_types[t + 1] else 1.0)
            return {
                'frames': images,
                'frame_labels': frame_labels,
                'inconsistency_labels': torch.tensor(inconsistency, dtype=torch.float32),
                'video_name': video_name,
            }

        if self.use_inconsistency_labels:
            # 从 frame_label.json 取帧类型，计算帧间不一致性（T-1）
            frame_label_path = os.path.join(video_path, 'frame_label.json')
            try:
                with open(frame_label_path, 'r', encoding='utf-8') as f:
                    frame_label_dict = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                T = images.size(0)
                return images, torch.zeros(1, dtype=torch.float32), torch.zeros(max(0, T - 1), dtype=torch.float32), video_name
            # 按采样顺序取每帧类型（key 为文件名无后缀，如 "0000"）
            frame_types = []
            for i in range(images.size(0)):
                idx = sample_indices[i].item()
                key = os.path.splitext(frame_files[idx])[0]
                if len(key) < 4:
                    key = key.zfill(4)
                frame_types.append(frame_label_dict.get(key, 'OR'))
            inconsistency = []
            for t in range(len(frame_types) - 1):
                inconsistency.append(0 if frame_types[t] == frame_types[t + 1] else 1)
            video_label = 1.0 if sum(inconsistency) > 0 else 0.0
            return (
                images,
                torch.tensor([video_label], dtype=torch.float32),
                torch.tensor(inconsistency, dtype=torch.float32),
                video_name,
            )
        return images, torch.tensor(labels, dtype=torch.float32), video_name

    def _temporal_augment(self, frames: torch.Tensor) -> torch.Tensor:
        """
        对训练集进行简单的时间增强：
        1. 随机局部扰动索引（jitter）
        2. 随机打乱小窗口
        3. 随机 temporal drop（以相邻帧替换）
        """
        num_frames = frames.size(0)
        if num_frames < 2:
            return frames

        # 1) jitter: 对索引做随机偏移并截断到合法范围
        jitter_strength = max(1, num_frames // 16)
        base_indices = torch.arange(num_frames)
        jitter = torch.randint(-jitter_strength, jitter_strength + 1, (num_frames,))
        jittered_indices = torch.clamp(base_indices + jitter, 0, num_frames - 1)
        frames = frames[jittered_indices]

        # 2) shuffle 小窗口
        if num_frames >= 3 and random.random() < 0.4:
            window_size = random.randint(2, min(4, num_frames))
            start = random.randint(0, num_frames - window_size)
            window = frames[start:start + window_size]
            perm = torch.randperm(window_size)
            frames[start:start + window_size] = window[perm]

        # 3) temporal drop：随机选择若干帧替换为邻帧
        if num_frames >= 2 and random.random() < 0.3:
            drop_count = max(1, num_frames // 10)
            for _ in range(drop_count):
                idx = random.randint(0, num_frames - 1)
                ref = idx - 1 if idx > 0 else idx + 1
                frames[idx] = frames[ref]

        return frames


def load_data(root_dir, label_names, split='train', transform=None, frames_per_video=16,
              use_inconsistency_labels=False, dual_task=False):
    return VideoDataset(
        root_dir, label_names, split, transform, frames_per_video,
        use_inconsistency_labels=use_inconsistency_labels,
        dual_task=dual_task,
    )


def collate_dual_task(batch):
    """双头任务 batch：list of dict -> dict of stacked tensors + list of names."""
    from torch.utils.data.dataloader import default_collate
    if not batch:
        return {}
    if isinstance(batch[0], dict):
        keys = batch[0].keys()
        out = {}
        for k in keys:
            if k == 'video_name':
                out[k] = [b[k] for b in batch]
            else:
                out[k] = torch.stack([b[k] for b in batch], dim=0)
        return out
    return default_collate(batch)


def get_label_names(root_dir, split='train', use_inconsistency_labels=False, dual_task=False):
    """
    获取数据集的标签名列表。
    dual_task=True 时返回 FH_LABEL_NAMES（6 类，帧级来源）。
    use_inconsistency_labels=True 且 Phase1 格式时，返回 [INCONSISTENCY_LABEL_NAME]（num_classes=1）。
    否则 Phase1 返回 FH_LABEL_NAMES（6 类）；按标签分文件夹则返回子目录名。
    """
    split_dir = os.path.join(root_dir, split)
    if not os.path.isdir(split_dir):
        return []
    if VideoDataset._is_phase1_format(split_dir):
        if dual_task:
            return list(FH_LABEL_NAMES)
        if use_inconsistency_labels:
            return [INCONSISTENCY_LABEL_NAME]
        return list(FH_LABEL_NAMES)
    return sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith('.')
    ])

