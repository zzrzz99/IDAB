"""
Video frame mixing utility for generating hybrid videos by partially replacing frames
between category pairs.

This script follows the workflow described by the user:

1. Iterate over every ordered pair of categories (A, B).
2. For each video in category A, find matching variants in category B using
   configurable matching strategies.
3. Replace selected frame indices (odd/even/every-n/custom/random) in the
   source video with frames from the matching target variant.
4. Save the mixed sequence into category-specific output folders for both A and B.
5. Record metadata describing the mixing process.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import re
DEFAULT_ABBREV_MAP: Dict[str, str] = {
    "Original": "Or",
    "Deepfakes": "DF",
    "Face2Face": "F2F",
    "FaceShifter": "FSH",
    "FaceSwap": "FSW",
    "NeuralTextures": "NT",
}

PAIR_ID_PATTERN = re.compile(r"(?:.+_)?([0-9]+_[0-9]+)$")



def setup_logger(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def sanitize_name(name: str) -> str:
    """Replace spaces with hyphens to keep folder names filesystem friendly."""
    return name.replace(" ", "-")


def load_mapping_file(mapping_path: Path) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
    """
    Load a CSV/JSON mapping file that explicitly specifies which videos should be paired.

    Returns:
        dict keyed by (cat_a, video_a) returning a list[(cat_b, video_b)]
    """
    if mapping_path.suffix.lower() == ".csv":
        pairs: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
        with mapping_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"category_a", "video_a", "category_b", "video_b"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"CSV mapping must contain columns: {required}")
            for row in reader:
                key = (row["category_a"], row["video_a"])
                pairs[key].append((row["category_b"], row["video_b"]))
        return pairs
    if mapping_path.suffix.lower() == ".json":
        with mapping_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        pairs: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
        for entry in raw:
            key = (entry["category_a"], entry["video_a"])
            pairs[key].append((entry["category_b"], entry["video_b"]))
        return pairs
    raise ValueError("Mapping file must be CSV or JSON.")


def list_video_dirs(category_dir: Path) -> List[Path]:
    return sorted(
        [d for d in category_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )


def get_prefix(name: str) -> str:
    """Extract prefix before the first underscore; fall back to the full name."""
    return name.split("_", 1)[0]


def compute_replacement_indices(
    total_frames: int,
    mode: str,
    *,
    index_base: int = 0,
    every_n: Optional[int] = None,
    custom_list: Optional[Sequence[int]] = None,
    random_p: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> Set[int]:
    """
    Determine which frame indices should be replaced.

    Args:
        total_frames: number of frames in the source video.
        mode: replacement strategy.
        index_base: whether indices are 0-based or 1-based for odd/even modes.
    """
    if total_frames <= 0:
        return set()

    indices: Set[int] = set()
    rng = random.Random(random_seed)

    if mode == "odd":
        for idx in range(total_frames):
            if (idx + index_base) % 2 == 1:
                indices.add(idx)
    elif mode == "even":
        for idx in range(total_frames):
            if (idx + index_base) % 2 == 0:
                indices.add(idx)
    elif mode == "every_n":
        if not every_n or every_n <= 0:
            raise ValueError("--every-n must be > 0 when mode=every_n")
        for idx in range(index_base, total_frames, every_n):
            if idx >= 0:
                indices.add(idx)
    elif mode == "custom_list":
        if custom_list is None:
            raise ValueError("--custom-list must be provided when mode=custom_list")
        for raw in custom_list:
            idx = raw - index_base if index_base == 1 else raw
            if 0 <= idx < total_frames:
                indices.add(idx)
    elif mode == "random_p":
        if random_p is None or not (0.0 < random_p <= 1.0):
            raise ValueError("--random-p must be in (0, 1] when mode=random_p")
        for idx in range(total_frames):
            if rng.random() < random_p:
                indices.add(idx)
    else:
        raise ValueError(f"Unsupported replacement mode: {mode}")

    return indices


def write_metadata(path: Path, metadata: Dict) -> None:
    metadata_path = path / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


@dataclass
class MatchResult:
    category_a: str
    video_a: Path
    category_b: str
    video_b: Path


def find_matches_for_video(
    video_a: Path,
    category_a: str,
    category_b: str,
    videos_b: List[Path],
    *,
    match_strategy: str,
) -> List[MatchResult]:
    matches: List[MatchResult] = []
    for video_b in videos_b:
        if match_strategy == "prefix":
            if get_prefix(video_b.name) != get_prefix(video_a.name):
                continue
        elif match_strategy == "exact":
            if video_b.name != video_a.name:
                continue
        else:
            raise ValueError(f"Unsupported match strategy: {match_strategy}")
        matches.append(
            MatchResult(
                category_a=category_a,
                video_a=video_a,
                category_b=category_b,
                video_b=video_b,
            )
        )
    return matches


def mix_frames(
    video_a: Path,
    video_b: Path,
    indices_to_replace: Set[int],
    *,
    truncate: bool = True,
    index_base: int = 0,
) -> Tuple[List[str], List[int]]:
    """
    Prepare frame source list for the mixed video.

    Returns:
        (frame_sources, applied_indices)
        frame_sources: list of source file paths (strings) to copy in order.
        applied_indices: the indices where replacement actually happened.
    """
    frames_a = sorted([p for p in video_a.iterdir() if p.is_file()])
    frames_b = sorted([p for p in video_b.iterdir() if p.is_file()])

    if not frames_a:
        raise RuntimeError(f"No frames found in {video_a}")
    if not frames_b:
        raise RuntimeError(f"No frames found in {video_b}")

    total_frames = len(frames_a)
    if truncate:
        max_replacement = min(len(frames_b), total_frames)
    else:
        max_replacement = len(frames_b)

    applied_indices: List[int] = []
    frame_sources: List[str] = []

    for idx in range(total_frames):
        if idx in indices_to_replace and idx < max_replacement:
            frame_sources.append(str(frames_b[idx]))
            applied_indices.append(idx)
        else:
            frame_sources.append(str(frames_a[idx]))

    return frame_sources, applied_indices


def copy_frames(frame_sources: Sequence[str], destination_dir: Path) -> None:
    for src in frame_sources:
        src_path = Path(src)
        dst_path = destination_dir / src_path.name
        shutil.copy2(src_path, dst_path)


def extract_pair_id(folder_name: str) -> Optional[str]:
    match = PAIR_ID_PATTERN.match(folder_name)
    if match:
        return match.group(1)
    return None


def format_mix_names(
    category_a: str,
    category_b: str,
    pair_id: str,
    *,
    abbrev_map: Dict[str, str],
) -> Tuple[str, str]:
    """
    Create folder names for the mixed outputs (A->B and B->A views).
    """
    def abbr(cat: str) -> str:
        if cat in abbrev_map:
            return abbrev_map[cat]
        parts = cat.replace("-", " ").split()
        if len(parts) == 1:
            token = parts[0]
            return "".join(token[:2]).title()
        return "".join(p[0].upper() for p in parts)

    a_tag = abbr(category_a)
    b_tag = abbr(category_b)
    mix_name_a = f"{a_tag}+{b_tag}_{pair_id}"
    mix_name_b = f"{b_tag}+{a_tag}_{pair_id}"
    return mix_name_a, mix_name_b


def prepare_output_dir(base_dir: Path) -> Optional[Path]:
    """
    Ensure output directory exists and has no existing files.
    If non-empty directory already exists, returns None to signal skipping.
    """
    if base_dir.exists():
        if any(base_dir.iterdir()):
            logging.info("Output directory %s already populated. Skipping.", base_dir)
            return None
    else:
        base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def process_pair(
    match: MatchResult,
    output_root: Path,
    *,
    indices_to_replace: Set[int],
    strategy_name: str,
    applied_strategy_params: Dict[str, object],
    abbrev_map: Optional[Dict[str, str]] = None,
    truncate: bool = True,
    attention_notes: Optional[str] = None,
) -> None:
    frame_sources, applied_indices = mix_frames(
        match.video_a,
        match.video_b,
        indices_to_replace,
        truncate=truncate,
    )

    pair_id = extract_pair_id(match.video_b.name)
    if pair_id is None:
        logging.warning("Unable to parse pair id from %s; using raw name.", match.video_b.name)
        pair_id = match.video_b.name

    mix_name_a, mix_name_b = format_mix_names(
        match.category_a,
        match.category_b,
        pair_id,
        abbrev_map=abbrev_map,
    )

    base_a = output_root / sanitize_name(match.category_a) / mix_name_a
    base_b = output_root / sanitize_name(match.category_b) / mix_name_b

    dest_a = prepare_output_dir(base_a)
    dest_b = prepare_output_dir(base_b)

    if dest_a is None and dest_b is None:
        return
    if dest_a is None or dest_b is None:
        logging.warning(
            "Skipping pair %s/%s vs %s/%s because one output directory already exists.",
            match.category_a,
            match.video_a.name,
            match.category_b,
            match.video_b.name,
        )
        return

    copy_frames(frame_sources, dest_a)
    copy_frames(frame_sources, dest_b)

    metadata = {
        "source_category": match.category_a,
        "target_category": match.category_b,
        "source_video": match.video_a.name,
        "target_video": match.video_b.name,
        "replacement_strategy": strategy_name,
        "strategy_params": applied_strategy_params,
        "replacement_indices": [int(i) for i in applied_indices],
        "total_frames": len(frame_sources),
        "notes": attention_notes,
    }
    write_metadata(dest_a, metadata)
    write_metadata(dest_b, metadata)

    logging.info(
        "Created mixed video %s (and flipped copy) with %d replaced frames.",
        dest_a,
        len(applied_indices),
    )


def parse_custom_list(value: str) -> List[int]:
    try:
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    except ValueError as err:
        raise argparse.ArgumentTypeError("custom index list must be comma-separated integers") from err


def parse_abbrev_map(path: Optional[Path]) -> Dict[str, str]:
    mapping: Dict[str, str] = dict(DEFAULT_ABBREV_MAP)
    if not path:
        return mapping
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Abbreviation map must be a JSON object")
    mapping.update({str(k): str(v) for k, v in data.items()})
    return mapping


def main(args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate mixed videos by replacing frames between category pairs.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root directory containing category subfolders.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory for mixed outputs.")
    parser.add_argument("--categories", nargs="+", default=[
        "Original",
        "Deepfakes",
        "Face2Face",
        "FaceShifter",
        "FaceSwap",
        "NeuralTextures",
    ], help="List of category folder names to consider.")
    parser.add_argument(
        "--match-strategy",
        choices=["prefix", "exact"],
        default="prefix",
        help="Strategy for mapping videos across categories when mapping file is not provided.",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        help="Optional CSV/JSON file specifying explicit video pairings.",
    )
    parser.add_argument(
        "--replacement-mode",
        choices=["odd", "even", "every_n", "custom_list", "random_p"],
        default="odd",
        help="Frame replacement strategy.",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=None,
        help="Stride for every_n replacement mode.",
    )
    parser.add_argument(
        "--custom-list",
        type=parse_custom_list,
        default=None,
        help="Comma-separated indices for custom_list replacement mode.",
    )
    parser.add_argument(
        "--random-p",
        type=float,
        default=None,
        help="Probability for random_p replacement mode.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for random_p mode.",
    )
    parser.add_argument(
        "--index-base",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether frame numbering is 0-based or 1-based for odd/even/custom strategies.",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate replacement when target video is shorter (default).",
    )
    parser.add_argument(
        "--no-truncate",
        dest="truncate",
        action="store_false",
        help="Disable truncation (looping not implemented).",
    )
    parser.set_defaults(truncate=True)
    parser.add_argument(
        "--abbrev-map",
        type=Path,
        help="Optional JSON file mapping category names to abbreviations used in output folder names.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    parsed = parser.parse_args(args)
    setup_logger(parsed.verbose)

    dataset_root: Path = parsed.dataset_root
    output_root: Path = parsed.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    abbrev_map = parse_abbrev_map(parsed.abbrev_map)

    categories = [sanitize_name(cat) for cat in parsed.categories]
    category_dirs = {cat: dataset_root / cat for cat in categories}
    for cat, path in category_dirs.items():
        if not path.exists():
            logging.warning("Category %s not found at %s. Skipping.", cat, path)

    mapping_data = load_mapping_file(parsed.mapping_file) if parsed.mapping_file else {}

    videos_by_category: Dict[str, List[Path]] = {}
    for cat, path in category_dirs.items():
        if path.exists():
            videos_by_category[cat] = list_video_dirs(path)

    strategy_params = {
        "index_base": parsed.index_base,
        "every_n": parsed.every_n,
        "custom_list": parsed.custom_list,
        "random_p": parsed.random_p,
        "random_seed": parsed.random_seed,
    }

    for cat_a, cat_b in permutations(categories, 2):
        if cat_a not in videos_by_category or cat_b not in videos_by_category:
            continue

        videos_a = videos_by_category[cat_a]
        videos_b = videos_by_category[cat_b]

        for video_a in videos_a:
            matches: List[MatchResult]
            key = (cat_a, video_a.name)
            if key in mapping_data:
                matches = [
                    MatchResult(cat_a, video_a, cat_b, dataset_root / sanitize_name(cat_b) / vid_b)
                    for cat_b, vid_b in mapping_data[key]
                ]
            else:
                matches = find_matches_for_video(
                    video_a,
                    cat_a,
                    cat_b,
                    videos_b,
                    match_strategy=parsed.match_strategy,
                )

            if not matches:
                logging.debug("No matches for %s/%s against category %s", cat_a, video_a.name, cat_b)
                continue

            for match in matches:
                total_frames = len([p for p in match.video_a.iterdir() if p.is_file()])
                indices = compute_replacement_indices(
                    total_frames,
                    parsed.replacement_mode,
                    index_base=parsed.index_base,
                    every_n=parsed.every_n,
                    custom_list=parsed.custom_list,
                    random_p=parsed.random_p,
                    random_seed=parsed.random_seed,
                )

                try:
                    process_pair(
                        match,
                        output_root,
                        indices_to_replace=indices,
                        strategy_name=parsed.replacement_mode,
                        applied_strategy_params=strategy_params,
                        abbrev_map=abbrev_map,
                        truncate=parsed.truncate,
                    )
                except Exception as err:  # pylint: disable=broad-except
                    logging.exception(
                        "Failed to process pair %s/%s with %s/%s: %s",
                        match.category_a,
                        match.video_a.name,
                        match.category_b,
                        match.video_b.name,
                        err,
                    )


if __name__ == "__main__":
    main()

