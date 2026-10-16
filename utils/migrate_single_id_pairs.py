"""
Utility to migrate single-id mixed folders (e.g. Or+DF_000) into the split
directories based on an existing split_map.json and rename them to the
corresponding two-id pattern (e.g. Or+DF_000_003).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence


SINGLE_ID_PATTERN = re.compile(r"(.+)_([0-9]{3})$")
DOUBLE_ID_PATTERN = re.compile(r"(.+)_([0-9]{3}_[0-9]{3})$")


def setup_logger(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def parse_split_map(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("split_map.json must be a JSON object mapping pair_id -> split")
    return {str(k): str(v) for k, v in data.items()}


def build_single_to_double_map(split_map: Dict[str, str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for pair_id in split_map.keys():
        parts = pair_id.split("_")
        if len(parts) != 2:
            continue
        a, b = parts
        mapping[a] = pair_id
        mapping[b] = f"{b}_{a}"
    return mapping


def ensure_unique_name(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path
    parent = base_path.parent
    stem = base_path.name
    suffix = 1
    while True:
        candidate = parent / f"{stem}_{suffix:02d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def canonical_pair_key(pair_id: str) -> str:
    """Sort the two ids so both orders map to the same key."""
    parts = pair_id.split("_")
    if len(parts) != 2:
        return pair_id
    return "_".join(sorted(parts))


def migrate_single_folder(
    src_folder: Path,
    single_id: str,
    *,
    pair_map: Dict[str, str],
    split_map: Dict[str, str],
    output_root: Path,
    dry_run: bool,
) -> None:
    match = SINGLE_ID_PATTERN.match(src_folder.name)
    if not match:
        logging.debug("Skipping %s (not single-id pattern).", src_folder)
        return
    prefix = match.group(1)
    single = match.group(2)

    if single not in pair_map:
        logging.warning("No mapping provided for single id %s (folder %s); skipping.", single, src_folder)
        return

    double_id = pair_map[single]
    if not DOUBLE_ID_PATTERN.match(f"{prefix}_{double_id}"):
        logging.warning("Mapped double id %s for %s does not fit expected pattern; skipping.", double_id, single)
        return

    split_key = canonical_pair_key(double_id)
    if split_key not in split_map:
        logging.warning("Mapped pair %s (canonical %s) not present in split_map; skipping %s.", double_id, split_key, src_folder)
        return

    split = split_map[split_key]

    category = src_folder.parent.name
    dst_folder_name = f"{prefix}_{double_id}"
    dst_folder = output_root / split / category / dst_folder_name
    dst_folder = ensure_unique_name(dst_folder)

    if dry_run:
        logging.info("[DRY-RUN] Would move %s -> %s", src_folder, dst_folder)
        return

    dst_folder.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Moving %s -> %s", src_folder, dst_folder)
    shutil.move(str(src_folder), str(dst_folder))


def main(args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Migrate single-id mixed folders using existing split map.")
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory containing category folders with single-id mixes.")
    parser.add_argument("--output-root", type=Path, required=True, help="Destination root (train/val/test already present).")
    parser.add_argument("--split-map", type=Path, required=True, help="Path to split_map.json generated during original split.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview operations without moving directories.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    parsed = parser.parse_args(args)
    setup_logger(parsed.verbose)

    if not parsed.split_map.exists():
        raise FileNotFoundError(f"split_map.json not found at {parsed.split_map}")

    split_map = parse_split_map(parsed.split_map)
    pair_map = build_single_to_double_map(split_map)

    categories = [p for p in parsed.input_root.iterdir() if p.is_dir()]

    for category_path in categories:
        for folder in category_path.iterdir():
            if not folder.is_dir():
                continue
            # only handle single-id pattern
            if not SINGLE_ID_PATTERN.match(folder.name):
                continue

            single_id = SINGLE_ID_PATTERN.match(folder.name).group(2)  # type: ignore
            migrate_single_folder(
                folder,
                single_id,
                pair_map=pair_map,
                split_map=split_map,
                output_root=parsed.output_root,
                dry_run=parsed.dry_run,
            )

    logging.info("Migration completed.")


if __name__ == "__main__":
    main()

