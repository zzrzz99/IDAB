"""
Copy single-forgery videos into split directories based on an existing split map.

For datasets containing original single forgery folders (e.g., category/000),
this utility moves/copies them into the split structure (train/val/test) using
the canonical pair key derived from split_map.json. No renaming is performed.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional


def setup_logger(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def load_split_map(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("split_map.json must be a JSON object mapping pair_id -> split")
    return {str(k): str(v) for k, v in data.items()}


def canonical_pair_key(pair_id: str) -> str:
    parts = pair_id.split("_")
    if len(parts) != 2:
        return pair_id
    return "_".join(sorted(parts))


def build_pair_lookup(split_map: Dict[str, str]) -> Dict[str, str]:
    """
    Build mapping from folder name (assumed to match pair_id) to split.
    """
    return {key: split for key, split in split_map.items()}


def find_single_folders(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def target_dir(output_root: Path, split: str, category: str, folder_name: str) -> Path:
    return output_root / split / category / folder_name


def ensure_unique(target: Path) -> Path:
    if not target.exists():
        return target
    parent = target.parent
    stem = target.name
    suffix = 1
    while True:
        candidate = parent / f"{stem}_{suffix:02d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def move_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(src, dst)
    else:
        shutil.move(str(src), str(dst))


def main(args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Copy/move single-forgery folders according to split_map.json.")
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory containing category subfolders.")
    parser.add_argument("--output-root", type=Path, required=True, help="Split output root (train/val/test).")
    parser.add_argument("--categories", nargs="+", help="Optional list of categories to process; defaults to all.")
    parser.add_argument("--split-map", type=Path, required=True, help="Path to split_map.json generated during pair split.")
    parser.add_argument("--copy", action="store_true", help="Copy instead of move.")
    parser.add_argument("--dry-run", action="store_true", help="Preview operations without moving/copying.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    parsed = parser.parse_args(args)
    setup_logger(parsed.verbose)

    input_root: Path = parsed.input_root
    output_root: Path = parsed.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    split_map = load_split_map(parsed.split_map)
    lookup = build_pair_lookup(split_map)

    categories = parsed.categories or sorted([p.name for p in input_root.iterdir() if p.is_dir()])
    logging.info("Processing categories: %s", ", ".join(categories))

    for category in categories:
        category_path = input_root / category
        if not category_path.exists():
            logging.warning("Category %s not found at %s; skipping.", category, category_path)
            continue

        for folder in find_single_folders(category_path):
            folder_name = folder.name
            canonical_key = canonical_pair_key(folder_name)
            if canonical_key not in lookup:
                logging.debug("Folder %s/%s not found in split map; skipping.", category, folder_name)
                continue

            split = lookup[canonical_key]
            dest = target_dir(output_root, split, category, folder_name)
            dest = ensure_unique(dest)

            if parsed.dry_run:
                logging.info("[DRY-RUN] Would %s %s -> %s", "copy" if parsed.copy else "move", folder, dest)
                continue

            logging.info("%s folder %s -> %s", "Copying" if parsed.copy else "Moving", folder, dest)
            move_or_copy(folder, dest, copy=parsed.copy)

    logging.info("Single-forgery migration completed.")


if __name__ == "__main__":
    main()

