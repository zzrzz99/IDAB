"""
Dataset splitter for mixed video pairs.

Transform a directory organised as:

    input_root/
        category_a/
            CATa+CATb_pairid/
        category_b/
            CATb+CATa_pairid/

Into train/val/test splits under output_root/, ensuring that all folders sharing
the same pair identifier (e.g. 000_003) end up in the same split.

Key features:
    * deterministic 70/20/10 split with configurable seed/ratios
    * dry-run mode for inspection before moving
    * moves directories (no copying) while preserving category hierarchy
    * logs operations and writes split manifests (JSON + CSV)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PAIR_REGEX = re.compile(r".+_([0-9]+_[0-9]+)$")
DEFAULT_SPLIT = (0.7, 0.2, 0.1)
DEFAULT_SEED = 42


@dataclass(frozen=True)
class PairEntry:
    category: str
    folder: str
    path: Path
    pair_id: str  # original order as in folder name


def setup_logger(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def extract_pair_id(folder_name: str) -> Optional[str]:
    match = PAIR_REGEX.match(folder_name)
    return match.group(1) if match else None


def canonical_pair_key(pair_id: str) -> str:
    parts = pair_id.split("_")
    if len(parts) != 2:
        return pair_id
    a, b = parts
    return "_".join(sorted([a, b]))


def build_pair_mapping(input_root: Path, categories: Sequence[str]) -> Dict[str, List[PairEntry]]:
    mapping: Dict[str, List[PairEntry]] = {}
    for category in categories:
        category_path = input_root / category
        if not category_path.exists():
            logging.warning("Category %s not found at %s; skipping.", category, category_path)
            continue
        for child in sorted(category_path.iterdir()):
            if not child.is_dir():
                continue
            pair_id = extract_pair_id(child.name)
            if not pair_id:
                logging.debug("Skipping %s/%s (no pair id detected).", category, child.name)
                continue
            key = canonical_pair_key(pair_id)
            mapping.setdefault(key, []).append(PairEntry(category, child.name, child, pair_id))
    return mapping


def determine_splits(
    pair_keys: Sequence[str],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, str]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    pairs = list(pair_keys)
    rnd = random.Random(seed)
    rnd.shuffle(pairs)

    n = len(pairs)
    train_count = int(n * ratios[0])
    val_count = int(n * ratios[1])
    test_count = n - train_count - val_count
    # adjust in case rounding removed all pairs
    if train_count == 0 and n > 0:
        train_count = min(1, n)
    if train_count + val_count > n:
        val_count = max(0, n - train_count)
    test_count = n - train_count - val_count

    split_map: Dict[str, str] = {}
    for idx, pair_id in enumerate(pairs):
        if idx < train_count:
            split_map[pair_id] = "train"
        elif idx < train_count + val_count:
            split_map[pair_id] = "val"
        else:
            split_map[pair_id] = "test"

    logging.info(
        "Split counts | train: %d | val: %d | test: %d | total pairs: %d",
        train_count,
        val_count,
        test_count,
        n,
    )
    return split_map


def target_path(output_root: Path, split: str, entry: PairEntry) -> Path:
    return output_root / split / entry.category / entry.folder


def ensure_target_ready(target: Path) -> bool:
    """
    Ensure destination directory is ready for move.
    Returns False if existing non-empty directory detected.
    """
    if target.exists():
        if any(target.iterdir()):
            logging.info("Target %s already exists with contents, skipping.", target)
            return False
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    return True


def move_pair(
    split: str,
    entries: Sequence[PairEntry],
    output_root: Path,
    *,
    dry_run: bool,
) -> List[Tuple[Path, Path]]:
    operations: List[Tuple[Path, Path]] = []
    for entry in entries:
        dest = target_path(output_root, split, entry)
        if not ensure_target_ready(dest):
            logging.warning(
                "Skipping pair because destination already populated: %s -> %s",
                entry.path,
                dest,
            )
            return []
        operations.append((entry.path, dest))

    if dry_run:
        for src, dst in operations:
            logging.info("[DRY-RUN] Would move %s -> %s", src, dst)
        return operations

    completed: List[Tuple[Path, Path]] = []
    try:
        for src, dst in operations:
            logging.debug("Moving %s -> %s", src, dst)
            shutil.move(str(src), str(dst))
            completed.append((src, dst))
    except Exception as exc:  # pylint: disable=broad-except
        logging.error("Error while moving %s -> %s : %s", src, dst, exc)
        # Attempt best-effort rollback for already moved directories
        for moved_src, moved_dst in reversed(completed):
            original_parent = moved_src.parent
            original_parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(moved_dst), str(original_parent / moved_dst.name))
        raise
    return operations


def write_manifests(
    outfile_dir: Path,
    records: List[Dict[str, str]],
    split_map: Dict[str, str],
) -> None:
    outfile_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = outfile_dir / "split_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "pair_key",
                "pair_id",
                "split",
                "category",
                "folder",
                "src_path",
                "dst_path",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    manifest_json = outfile_dir / "split_map.json"
    with manifest_json.open("w", encoding="utf-8") as f_json:
        json.dump(split_map, f_json, indent=2, ensure_ascii=False)


def parse_categories(value: Optional[Sequence[str]], input_root: Path) -> List[str]:
    if value:
        return [str(v) for v in value]
    return sorted([p.name for p in input_root.iterdir() if p.is_dir()])


def parse_ratios(train: float, val: float, test: float) -> Tuple[float, float, float]:
    ratios = (train, val, test)
    if not all(r >= 0 for r in ratios):
        raise ValueError("Split ratios must be non-negative")
    total = sum(ratios)
    if total == 0:
        raise ValueError("Split ratios cannot all be zero")
    ratios = tuple(r / total for r in ratios)  # normalise to sum 1
    return ratios  # type: ignore


def main(args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Split mixed dataset directories into train/val/test partitions.")
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory containing category subfolders.")
    parser.add_argument("--output-root", type=Path, required=True, help="Destination root directory.")
    parser.add_argument("--categories", nargs="+", help="Optional list of categories to process. Defaults to all.")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT[0], help="Train split ratio (default 0.7).")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_SPLIT[1], help="Validation split ratio (default 0.2).")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT[2], help="Test split ratio (default 0.1).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for split reproducibility.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned operations without moving any files.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        help="Directory to write manifest files (defaults to output-root if omitted).",
    )

    parsed = parser.parse_args(args)
    setup_logger(parsed.verbose)

    input_root: Path = parsed.input_root
    output_root: Path = parsed.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    categories = parse_categories(parsed.categories, input_root)
    logging.info("Processing categories: %s", ", ".join(categories))

    pair_mapping = build_pair_mapping(input_root, categories)
    total_pairs = len(pair_mapping)
    logging.info("Detected %d unique pair ids.", total_pairs)
    if total_pairs == 0:
        logging.warning("No pairs found. Exiting.")
        return

    ratios = parse_ratios(parsed.train_ratio, parsed.val_ratio, parsed.test_ratio)
    split_map = determine_splits(pair_mapping.keys(), ratios, parsed.seed)

    if parsed.dry_run:
        logging.info("[DRY-RUN] No directories will be moved.")

    manifest_records: List[Dict[str, str]] = []

    for pair_key, entries in pair_mapping.items():
        split = split_map[pair_key]
        logging.debug("Pair %s assigned to %s (%d entries).", pair_key, split, len(entries))

        operations = move_pair(split, entries, output_root, dry_run=parsed.dry_run)
        if not operations:
            continue
        if parsed.dry_run:
            continue

        for entry, (src, dst) in zip(entries, operations):
            manifest_records.append(
                {
                    "pair_key": pair_key,
                    "pair_id": entry.pair_id,
                    "split": split,
                    "category": dst.parent.name,
                    "folder": dst.name,
                    "src_path": str(src),
                    "dst_path": str(dst),
                }
            )

    manifest_dir = parsed.manifest_dir or output_root
    write_manifests(manifest_dir, manifest_records, split_map)

    logging.info("Completed dataset split. Manifests written to %s", manifest_dir)


if __name__ == "__main__":
    main()

