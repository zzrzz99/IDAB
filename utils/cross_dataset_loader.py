"""
跨数据集泛化测试用的二分类视频数据集。

适配两种「标签在路径上」的已裁剪帧数据集：

Celeb-DF（celeb-1）:
    root/Celeb-real/id0_0000/000000.jpg          -> real
    root/YouTube-real/00000/000000.jpg            -> real
    root/Celeb-synthesis/id0_id1_0000/000000.jpg  -> fake

DFDC（DFDC-1）:
    root/DFDC_REAL_Face_only_data/<id>/000000.jpg -> real
    root/DFDC_FAKE_Face_only_data/<id>/000000.jpg -> fake

规则（大小写不敏感，可通过参数覆盖）：
    相对路径中命中 fake 关键字（'fake'、'synthesis'）-> 1 (fake)
    相对路径中命中 real 关键字（'real'）             -> 0 (real)

每个「直接包含图片的目录」视为一个视频样本，帧采样与训练侧一致
（按文件名排序 + linspace 均匀重采样到固定帧数）。
"""
import os
import logging
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
DEFAULT_REAL_KEYWORDS = ('real',)
DEFAULT_FAKE_KEYWORDS = ('fake', 'synthesis')


def classify_path(rel_path, real_keywords=DEFAULT_REAL_KEYWORDS,
                  fake_keywords=DEFAULT_FAKE_KEYWORDS):
    """根据相对路径返回二分类标签：0=real, 1=fake, None=无法识别。"""
    p = rel_path.replace('\\', '/').lower()
    # fake 优先（个别目录名可能同时含 real/fake 子串）
    for kw in fake_keywords:
        if kw.lower() in p:
            return 1
    for kw in real_keywords:
        if kw.lower() in p:
            return 0
    return None


def scan_binary_videos(root, real_keywords=DEFAULT_REAL_KEYWORDS,
                       fake_keywords=DEFAULT_FAKE_KEYWORDS, min_frames=1):
    """
    递归扫描 root：凡「直接包含图片文件」的目录记为一个视频样本。
    返回 [(video_dir, label, rel_dir_name), ...]，按路径排序（确定性顺序）。
    """
    samples = []
    skipped = 0
    for dirpath, dirnames, filenames in os.walk(root):
        imgs = [f for f in filenames if f.lower().endswith(IMG_EXTS)]
        if not imgs:
            continue
        rel = os.path.relpath(dirpath, root)
        label = classify_path(rel, real_keywords, fake_keywords)
        if label is None:
            skipped += 1
            continue
        if len(imgs) < min_frames:
            skipped += 1
            continue
        samples.append((dirpath, label, rel.replace('\\', '/')))
    samples.sort(key=lambda s: s[2])
    if skipped:
        logging.warning('scan_binary_videos: 跳过无法判定/帧数不足的目录 %d 个', skipped)
    return samples


class BinaryVideoFrameDataset(Dataset):
    """
    路径二分类视频数据集。
    返回 (frames[T, C, H, W], label(tensor scalar float, 0=real/1=fake), video_name: str)。
    """

    def __init__(self, root, transform=None, frames_per_video=12,
                 real_keywords=DEFAULT_REAL_KEYWORDS,
                 fake_keywords=DEFAULT_FAKE_KEYWORDS,
                 samples=None, max_samples_per_class=None, seed=42):
        self.root = root
        self.transform = transform
        self.frames_per_video = frames_per_video
        if samples is None:
            samples = scan_binary_videos(root, real_keywords, fake_keywords)
        if max_samples_per_class is not None:
            import random
            rng = random.Random(seed)
            by_label = {0: [], 1: []}
            for s in samples:
                by_label[s[1]].append(s)
            picked = []
            for lab, lst in by_label.items():
                if len(lst) > max_samples_per_class:
                    picked.extend(rng.sample(lst, max_samples_per_class))
                else:
                    picked.extend(lst)
            picked.sort(key=lambda s: s[2])
            samples = picked
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, video_dir):
        files = sorted(
            f for f in os.listdir(video_dir) if f.lower().endswith(IMG_EXTS)
        )
        n = len(files)
        if n == 0:
            return []
        if n == 1 and self.frames_per_video > 1:
            idx = [0] * self.frames_per_video
        elif n != self.frames_per_video:
            idx = torch.linspace(0, n - 1, steps=self.frames_per_video,
                                 dtype=torch.float32).round().long()
            idx = torch.clamp(idx, 0, n - 1).tolist()
        else:
            idx = list(range(n))
        frames = []
        for i in idx:
            fp = os.path.join(video_dir, files[i])
            try:
                img = Image.open(fp).convert('RGB')
            except (UnidentifiedImageError, OSError):
                continue
            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)
        return frames

    def __getitem__(self, index):
        video_dir, label, name = self.samples[index]
        frames = self._load_frames(video_dir)
        if not frames:
            logging.warning('视频全部帧损坏 %s，返回零张量', video_dir)
            frames = [torch.zeros(3, 256, 256, dtype=torch.float32)
                      for _ in range(self.frames_per_video)]
        if len(frames) == 1 and self.frames_per_video > 1:
            frames = frames * self.frames_per_video
        elif len(frames) != self.frames_per_video:
            # 个别帧损坏导致数量不足：按 linspace 在已加载帧中重采样补齐
            src = torch.stack(frames)
            new_idx = torch.linspace(0, src.size(0) - 1, steps=self.frames_per_video,
                                     dtype=torch.float32).round().long()
            new_idx = torch.clamp(new_idx, 0, src.size(0) - 1)
            frames = list(src[new_idx])
        frames = torch.stack(frames)
        return frames, torch.tensor(float(label), dtype=torch.float32), name


def binary_collate(batch):
    frames = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    names = [b[2] for b in batch]
    return frames, labels, names


# 数据集预设：root 下第一层目录命名习惯
DATASET_PRESETS = {
    'celeb': {
        'dir_name': 'celeb-1',
        'real_keywords': ('celeb-real', 'youtube-real'),
        'fake_keywords': ('celeb-synthesis',),
    },
    'dfdc': {
        'dir_name': 'DFDC-1',
        'real_keywords': ('dfdc_real',),
        'fake_keywords': ('dfdc_fake',),
    },
}
