"""
原始 FF++ 裁剪帧数据集加载器（dataset-1 结构）。

适配目录结构（无 split 层，标签在方法目录名中）：
    root/Original/<id>/*.jpg                         真实帧
    root/Deepfakes/<id>_<partner>/*.jpg              伪造帧
    root/Face2Face/<id>_<partner>/*.jpg
    root/FaceShifter/<id>_<partner>/*.jpg
    root/FaceSwap/<id>_<partner>/*.jpg
    root/NeuralTextures/<id>_<partner>/*.jpg

划分规则（与 data/phase0_id_split.py 一致，保证身份隔离）：
    1. 从 root/Original 收集 sorted 身份 ID，按 7:2:1 切成 train/val/test（无随机）
    2. 支持传入 id_split.json（--id_split）复用 FH-FF++ 的同一份身份划分
    3. 伪造视频按文件夹名「第一个 ID」（目标身份）归属 split，与 phase1 一致
    4. 划分结果可导出 JSON（dump_split），便于多机复现同一份 split

标签：视频级单标签 multi-hot，类别顺序与现有训练框架一致
    ["DF", "F2F", "FS", "FSh", "NT", "OR"]，自动跳过 root 下缺失的方法目录。

__getitem__ 返回与 utils.dataset_loader.VideoDataset 普通模式完全一致：
    (frames[T, C, H, W], label_vec[num_classes], video_name)
"""
import os
import json
import random
import logging
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

# 与 utils/dataset_loader.py 的 FH_LABEL_NAMES 保持一致
FFPP_LABEL_ORDER = ["DF", "F2F", "FS", "FSh", "NT", "OR"]
# 方法目录名 -> 缩写（与 data/ffpp_config.py 一致）
METHOD_TO_ABBR = {
    "Original": "OR",
    "Deepfakes": "DF",
    "Face2Face": "F2F",
    "FaceShifter": "FSh",
    "FaceSwap": "FS",
    "NeuralTextures": "NT",
}
FRAME_EXTS = (".jpg", ".jpeg", ".png")
SPLIT_RATIO = (0.7, 0.2, 0.1)


def get_ffpp_label_names(root_dir):
    """
    扫描 root 下实际存在的方法目录，按固定顺序返回缩写列表。
    例如 6 个目录齐全 -> ['DF', 'F2F', 'FS', 'FSh', 'NT', 'OR']；缺 NT -> 5 类。
    """
    present = set()
    if os.path.isdir(root_dir):
        for d in os.listdir(root_dir):
            if d in METHOD_TO_ABBR and os.path.isdir(os.path.join(root_dir, d)):
                present.add(METHOD_TO_ABBR[d])
    return [abbr for abbr in FFPP_LABEL_ORDER if abbr in present]


def _collect_identity_ids(root_dir):
    """从 root/Original 收集 sorted 身份 ID（文件夹名 000/001/...）。"""
    original_dir = os.path.join(root_dir, "Original")
    if not os.path.isdir(original_dir):
        raise FileNotFoundError(f"未找到 Original 目录: {original_dir}")
    ids = []
    for name in os.listdir(original_dir):
        path = os.path.join(original_dir, name)
        if os.path.isdir(path) and not name.startswith("."):
            ids.append(name)
    return sorted(ids)


def _split_ids(ids, ratio=SPLIT_RATIO):
    """与 phase0_id_split.split_ids 相同的确定性切分（无随机）。"""
    train_r, val_r, test_r = ratio
    n = len(ids)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def build_identity_split(root_dir, id_split_path=None, ratio=SPLIT_RATIO):
    """
    返回 {"train": set, "val": set, "test": set, "raw": {...}}。
    优先从 JSON 加载（键 train/val/test），否则现场按 7:2:1 划分。
    """
    raw = None
    if id_split_path and os.path.isfile(id_split_path):
        with open(id_split_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        missing = [k for k in ("train", "val", "test") if k not in raw]
        if missing:
            raise KeyError(f"id_split.json 缺少键: {missing}")
        ids = raw.get("all_ids")
        if ids is None:
            ids = sorted(set(raw["train"]) | set(raw["val"]) | set(raw["test"]))
        logging.info("从 %s 加载身份划分", id_split_path)
    else:
        ids = _collect_identity_ids(root_dir)
        if not ids:
            raise ValueError("未收集到任何身份 ID，请检查 Original 目录")
        raw = _split_ids(ids, ratio)
        raw["all_ids"] = ids
        raw["total"] = len(ids)
        logging.info("未提供 id_split.json，按 sorted ID 7:2:1 现场划分（与 Phase0 一致）")
    return {
        "train": set(raw["train"]),
        "val": set(raw["val"]),
        "test": set(raw["test"]),
        "raw": raw,
    }


def dump_split(split_info, out_path):
    """把实际使用的划分写成 JSON，便于复现。"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    raw = split_info["raw"]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)


def _identity_of(method_abbr, video_folder):
    """
    视频文件夹名 -> 归属身份 ID：
      Original/000      -> "000"
      Deepfakes/000_003 -> "000"（第一个 ID，即被伪造的目标身份）
    解析失败返回 None。
    """
    if method_abbr == "OR":
        name = video_folder
    else:
        name = video_folder.split("_")[0]
    return name if name.isdigit() else None


class FFPPOriginalDataset(Dataset):
    """
    原始 FF++ 视频级单标签数据集。
    label_names: get_ffpp_label_names(root_dir) 的结果，全局一致，不要按 split 分别扫描。
    broadcast_frame_labels=True 时返回 4 元组 (frames, label_vec, frame_labels, video_name)，
    其中 frame_labels 为视频标签广播到每一帧的 (T, num_classes) 张量，
    用于 compare 双头模型的帧级监督（原始 FF++ 全视频同一来源，无帧间切换）。
    """

    def __init__(self, root_dir, label_names, split="train", transform=None,
                 frames_per_video=16, split_info=None, broadcast_frame_labels=False):
        self.root_dir = root_dir
        self.label_names = list(label_names)
        self.split = split
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.broadcast_frame_labels = broadcast_frame_labels
        if split_info is None:
            split_info = build_identity_split(root_dir)
        self.split_info = split_info
        self.samples = self._gather_samples()
        self._default_frame_shape = self._infer_default_shape()

    def _infer_default_shape(self):
        if self.transform:
            try:
                dummy = Image.new("RGB", (256, 256))
                return tuple(self.transform(dummy).shape)
            except Exception:
                pass
        return (3, 256, 256)

    def _gather_samples(self):
        allowed_ids = self.split_info[self.split]
        abbr_to_dir = {v: k for k, v in METHOD_TO_ABBR.items()}
        samples = []
        stats = {}
        for abbr in self.label_names:
            method_dir_name = abbr_to_dir[abbr]
            method_dir = os.path.join(self.root_dir, method_dir_name)
            if not os.path.isdir(method_dir):
                continue
            label_idx = self.label_names.index(abbr)
            n_kept = 0
            for video_folder in sorted(os.listdir(method_dir)):
                video_path = os.path.join(method_dir, video_folder)
                if not os.path.isdir(video_path) or video_folder.startswith("."):
                    continue
                identity = _identity_of(abbr, video_folder)
                if identity is None or identity not in allowed_ids:
                    continue
                frame_files = [
                    f for f in os.listdir(video_path)
                    if f.lower().endswith(FRAME_EXTS)
                ]
                if not frame_files:
                    logging.warning("跳过空帧目录: %s", video_path)
                    continue
                label_vec = [0] * len(self.label_names)
                label_vec[label_idx] = 1
                # 唯一视频名：缩写_文件夹名，避免不同方法同 id 目录重名
                unique_name = f"{abbr}_{video_folder}"
                samples.append((video_path, label_vec, unique_name))
                n_kept += 1
            stats[abbr] = n_kept
        logging.info("[%s] 样本数: %s, 合计 %d", self.split, stats, len(samples))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, labels, video_name = self.samples[idx]
        frame_files = sorted(
            [f for f in os.listdir(video_path) if f.lower().endswith(FRAME_EXTS)]
        )
        images = []
        for frame in frame_files:
            frame_path = os.path.join(video_path, frame)
            try:
                with Image.open(frame_path) as img:
                    img = img.convert("RGB")
                    if self.transform:
                        img = self.transform(img)
                    else:
                        img = transforms.ToTensor()(img)
                    images.append(img)
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as err:
                logging.warning("跳过损坏帧 %s: %s", frame_path, err)
                continue
        if not images:
            logging.warning("视频全部帧损坏 %s，返回零张量", video_path)
            C, H, W = self._default_frame_shape
            empty = torch.zeros(self.frames_per_video, C, H, W, dtype=torch.float32)
            label_t = torch.tensor(labels, dtype=torch.float32)
            if self.broadcast_frame_labels:
                return empty, label_t, label_t.unsqueeze(0).repeat(self.frames_per_video, 1), video_name
            return empty, label_t, video_name

        images = torch.stack(images)  # (valid_frames, C, H, W)
        if self.split == "train":
            images = self._temporal_augment(images)
        valid_frames = images.size(0)
        if valid_frames == 1 and self.frames_per_video > 1:
            images = images.repeat(self.frames_per_video, 1, 1, 1)
        elif valid_frames != self.frames_per_video:
            indices = torch.linspace(
                0, valid_frames - 1, steps=self.frames_per_video, dtype=torch.float32
            ).round().long()
            indices = torch.clamp(indices, 0, valid_frames - 1)
            images = images[indices]
        label_t = torch.tensor(labels, dtype=torch.float32)
        if self.broadcast_frame_labels:
            frame_labels = label_t.unsqueeze(0).repeat(images.size(0), 1)  # (T, num_classes)
            return images, label_t, frame_labels, video_name
        return images, label_t, video_name

    @staticmethod
    def _temporal_augment(frames: torch.Tensor) -> torch.Tensor:
        """与 VideoDataset._temporal_augment 相同：jitter + 小窗口 shuffle + temporal drop。"""
        num_frames = frames.size(0)
        if num_frames < 2:
            return frames
        jitter_strength = max(1, num_frames // 16)
        base_indices = torch.arange(num_frames)
        jitter = torch.randint(-jitter_strength, jitter_strength + 1, (num_frames,))
        frames = frames[torch.clamp(base_indices + jitter, 0, num_frames - 1)]
        if num_frames >= 3 and random.random() < 0.4:
            window_size = random.randint(2, min(4, num_frames))
            start = random.randint(0, num_frames - window_size)
            window = frames[start:start + window_size]
            frames[start:start + window_size] = window[torch.randperm(window_size)]
        if num_frames >= 2 and random.random() < 0.3:
            drop_count = max(1, num_frames // 10)
            for _ in range(drop_count):
                idx = random.randint(0, num_frames - 1)
                ref = idx - 1 if idx > 0 else idx + 1
                frames[idx] = frames[ref]
        return frames


def load_ffpp_original(root_dir, label_names, split="train", transform=None,
                       frames_per_video=16, split_info=None, broadcast_frame_labels=False):
    """与 utils.dataset_loader.load_data 对齐的工厂函数。"""
    return FFPPOriginalDataset(
        root_dir, label_names, split=split, transform=transform,
        frames_per_video=frames_per_video, split_info=split_info,
        broadcast_frame_labels=broadcast_frame_labels,
    )
