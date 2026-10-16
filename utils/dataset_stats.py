"""
Dataset statistics:
- total videos
- single-source vs heterogeneous videos
- total frames
- per-class frame counts
- train/val/test split availability

Supports Phase1 (FH-FF++) format:
  data_dir/split/video_xxxx/frames/*.jpg + frame_label.json

Usage:
  python utils/dataset_stats.py --data_dir /path/to/data-or-nt
  python utils/dataset_stats.py --data_dir /path/to/data --output dataset_stats.txt
"""
import os
import json
import argparse
from collections import defaultdict

# Consistent with dataset_loader
FH_LABEL_NAMES = ["DF", "F2F", "FS", "FSh", "NT", "OR"]


def _frame_dir(video_path):
    frames_sub = os.path.join(video_path, 'frames')
    if os.path.isdir(frames_sub):
        return frames_sub
    return video_path


def _is_phase1_format(split_dir):
    if not os.path.isdir(split_dir):
        return False
    for d in os.listdir(split_dir):
        if d.startswith('.') or not d.startswith('video_'):
            continue
        sub = os.path.join(split_dir, d)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, 'video_label.json')):
            return True
    return False


def _collect_phase1_split(split_dir, require_frame_label=True):
    """Collect video statistics for one split directory."""
    videos = []
    for video_name in sorted(os.listdir(split_dir)):
        if video_name.startswith('.') or not video_name.startswith('video_'):
            continue
        video_path = os.path.join(split_dir, video_name)
        if not os.path.isdir(video_path):
            continue
        frame_label_path = os.path.join(video_path, 'frame_label.json')
        if require_frame_label and not os.path.isfile(frame_label_path):
            continue
        frame_dir = _frame_dir(video_path)
        if not os.path.isdir(frame_dir):
            continue
        frame_files = [
            f for f in os.listdir(frame_dir)
            if f.lower().endswith(('jpg', 'jpeg', 'png'))
        ]
        if not frame_files:
            continue
        frame_label_dict = {}
        if os.path.isfile(frame_label_path):
            try:
                with open(frame_label_path, 'r', encoding='utf-8') as f:
                    frame_label_dict = json.load(f)
            except (json.JSONDecodeError, Exception):
                pass
        frame_types = []
        for f in sorted(frame_files):
            key = os.path.splitext(f)[0]
            if len(key) < 4:
                key = key.zfill(4)
            frame_types.append(frame_label_dict.get(key, 'OR'))
        unique_types = set(frame_types)
        is_single_source = len(unique_types) <= 1
        per_class = defaultdict(int)
        for ft in frame_types:
            if ft in FH_LABEL_NAMES:
                per_class[ft] += 1
            else:
                per_class['OR'] += 1
        videos.append({
            'name': video_name,
            'num_frames': len(frame_files),
            'frame_types': frame_types,
            'unique_types': unique_types,
            'is_single_source': is_single_source,
            'per_class': dict(per_class),
        })
    return videos


def run_stats(data_dir, require_frame_label=True):
    """Compute stats under data_dir for train/val/test."""
    splits = ['train', 'val', 'test']
    out = {
        'data_dir': os.path.abspath(data_dir),
        'splits': {},
        'total_videos': 0,
        'total_frames': 0,
        'single_source_videos': 0,
        'heterogeneous_videos': 0,
        'per_class_frames': defaultdict(int),
        'has_train_val_test': False,
    }
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            out['splits'][split] = {'present': False, 'videos': 0, 'frames': 0, 'single': 0, 'hetero': 0}
            continue
        if not _is_phase1_format(split_dir):
            out['splits'][split] = {'present': True, 'format': 'non_phase1', 'videos': 0, 'frames': 0}
            continue
        videos = _collect_phase1_split(split_dir, require_frame_label=require_frame_label)
        n_videos = len(videos)
        n_frames = sum(v['num_frames'] for v in videos)
        n_single = sum(1 for v in videos if v['is_single_source'])
        n_hetero = n_videos - n_single
        out['splits'][split] = {
            'present': True,
            'format': 'phase1',
            'videos': n_videos,
            'frames': n_frames,
            'single_source': n_single,
            'heterogeneous': n_hetero,
        }
        out['total_videos'] += n_videos
        out['total_frames'] += n_frames
        out['single_source_videos'] += n_single
        out['heterogeneous_videos'] += n_hetero
        for v in videos:
            for cls, cnt in v['per_class'].items():
                out['per_class_frames'][cls] += cnt
    out['per_class_frames'] = dict(out['per_class_frames'])
    out['has_train_val_test'] = (
        out['splits'].get('train', {}).get('present') and
        out['splits'].get('val', {}).get('present') and
        out['splits'].get('test', {}).get('present')
    )
    return out


def format_report(stats):
    """Format a human-readable report string."""
    lines = []
    lines.append('=' * 60)
    lines.append('Dataset Statistics Report')
    lines.append('=' * 60)
    lines.append(f'Data directory: {stats["data_dir"]}')
    lines.append('')
    lines.append('1) Splits')
    lines.append('-' * 40)
    for split in ['train', 'val', 'test']:
        s = stats['splits'].get(split, {})
        if s.get('present'):
            fmt = s.get('format', 'phase1')
            lines.append(f'  {split:5s}: present, format={fmt}, videos={s.get("videos", 0)}, frames={s.get("frames", 0)}')
            if fmt == 'phase1':
                lines.append(f'        single_source={s.get("single_source", 0)}, heterogeneous={s.get("heterogeneous", 0)}')
        else:
            lines.append(f'  {split:5s}: missing')
    lines.append(f'  Has train/val/test splits: {"yes" if stats["has_train_val_test"] else "no"}')
    lines.append('')
    lines.append('2) Overall')
    lines.append('-' * 40)
    lines.append(f'  Total videos:            {stats["total_videos"]}')
    lines.append(f'  Single-source videos:    {stats["single_source_videos"]}')
    lines.append(f'  Heterogeneous videos:    {stats["heterogeneous_videos"]}')
    lines.append(f'  Total frames:            {stats["total_frames"]}')
    lines.append('')
    lines.append('3) Per-class frame counts')
    lines.append('-' * 40)
    for name in FH_LABEL_NAMES:
        cnt = stats['per_class_frames'].get(name, 0)
        lines.append(f'  {name:6s}: {cnt}')
    other = sum(v for k, v in stats['per_class_frames'].items() if k not in FH_LABEL_NAMES)
    if other:
        lines.append(f'  Other:  {other}')
    lines.append('')
    lines.append('=' * 60)
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Phase1 dataset stats: videos, single/hetero, frames, per-class, splits')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset root (contains train/val/test)')
    parser.add_argument('--output', type=str, default=None, help='Output txt path (default: <data_dir>/dataset_stats.txt)')
    parser.add_argument('--no_frame_label', action='store_true', help='Do not require frame_label.json (only count videos with video_label)')
    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f'Error: directory does not exist: {data_dir}')
        return
    stats = run_stats(data_dir, require_frame_label=not args.no_frame_label)
    report = format_report(stats)
    print(report)
    out_path = args.output
    if out_path is None:
        out_path = os.path.join(data_dir, 'dataset_stats.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nReport saved: {out_path}')


if __name__ == '__main__':
    main()
