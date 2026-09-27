"""
build_mute_dataset.py

Convert the raw MUTE Excel files into the same standardized JSONL + images/
format expected by the rest of the MemeBLIP2 pipeline.

Expected raw structure:
    RAW_MUTE_ROOT/
      MUTE/
        train_hate.xlsx
        valid_hate.xlsx
        test_hate.xlsx
      ... image files somewhere under RAW_MUTE_ROOT ...

Expected Excel columns:
    image_name, Captions, Label

Label mapping:
    not-hate -> 0
    hate     -> 1

Outputs:
    DATASET_ROOT/dataset.jsonl
    DATASET_ROOT/dataset_unbalanced_metadata.jsonl
    DATASET_ROOT/train.jsonl
    DATASET_ROOT/val.jsonl
    DATASET_ROOT/test.jsonl
    DATASET_ROOT/images/

Important:
    This script only standardizes the raw MUTE dataset.
    prepare_dataset.py remains responsible for producing:
        processed_splits/train_rows.jsonl
        processed_splits/val_rows.jsonl
        processed_splits/test_rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

try:
    from config import (
        RAW_MUTE_ROOT,
        DATASET_ROOT,
        RANDOM_SEED,
        MUTE_BALANCE_SCOPE,
        MUTE_ID_PREFIX,
        MUTE_LABEL_MAP,
        ensure_project_dirs,
    )
except Exception:
    RAW_MUTE_ROOT = Path("/scratch/MUTE")
    DATASET_ROOT = Path("/scratch/meme_dataset_separated/mute")
    RANDOM_SEED = 42
    MUTE_BALANCE_SCOPE = "combined"
    MUTE_ID_PREFIX = "mute"
    MUTE_LABEL_MAP = {"not-hate": 0, "hate": 1}

    def ensure_project_dirs():
        DATASET_ROOT.mkdir(parents=True, exist_ok=True)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EXCEL_SPLIT_FILES = {
    "train": "train_hate.xlsx",
    "val": "valid_hate.xlsx",
    "test": "test_hate.xlsx",
}
REQUIRED_EXCEL_COLUMNS = ["image_name", "Captions", "Label"]


# ============================================================
# 1. Basic helpers
# ============================================================

def normalize_label(value) -> int:
    """
    Normalize raw MUTE label strings to binary labels.
    """
    label = str(value).strip().lower()

    if label in MUTE_LABEL_MAP:
        return int(MUTE_LABEL_MAP[label])

    # Small tolerance for common variants.
    aliases = {
        "non-hate": 0,
        "nonhate": 0,
        "not_hate": 0,
        "not hate": 0,
        "0": 0,
        "hateful": 1,
        "hate": 1,
        "1": 1,
    }

    if label in aliases:
        return aliases[label]

    raise ValueError(f"Unknown label: {value!r}")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_source_name(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("/", "_").replace("\\", "_")
    return value or "mute"


# ============================================================
# 2. Image indexing
# ============================================================

def find_image_index(raw_root: Path) -> Tuple[Dict[str, Path], int]:
    """
    Build a case-insensitive filename -> absolute path index for images.

    If duplicate filenames exist, the first discovered image is kept and the
    duplicate count is reported. This matches the previous behavior while making
    the ambiguity visible.
    """
    index: Dict[str, Path] = {}
    duplicate_count = 0

    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        key = path.name.lower()
        if key in index:
            duplicate_count += 1
            continue
        index[key] = path

    return index, duplicate_count


# ============================================================
# 3. Excel reading
# ============================================================

def read_split(
    split_name: str,
    file_path: Path,
    image_index: Dict[str, Path],
) -> Tuple[List[dict], int, int, int]:
    """
    Read one MUTE Excel file and return standardized internal row dictionaries.

    Internal keys prefixed with "_" are removed before writing JSONL.
    """
    df = pd.read_excel(file_path)

    missing_cols = [c for c in REQUIRED_EXCEL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{file_path} missing columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    rows: List[dict] = []
    missing_images = 0
    invalid_labels = 0
    empty_image_names = 0

    for i, row in df.iterrows():
        image_name = clean_text(row["image_name"])
        if not image_name:
            empty_image_names += 1
            continue

        text = clean_text(row["Captions"])

        try:
            label = normalize_label(row["Label"])
        except Exception:
            invalid_labels += 1
            continue

        src_image = image_index.get(image_name.lower())
        if src_image is None:
            missing_images += 1
            continue

        rows.append({
            "_src_image_abs": str(src_image),
            "_source_row_index": int(i),
            "text": text,
            "label": int(label),
            "source": "mute",
            "original_split": split_name,
            "original_image_name": image_name,
        })

    return rows, missing_images, invalid_labels, empty_image_names


def load_all_raw_rows(raw_root: Path) -> Tuple[List[dict], dict]:
    """
    Load train/val/test Excel files and image paths from RAW_MUTE_ROOT.
    """
    mute_excel_dir = raw_root / "MUTE"
    split_files = {
        split: mute_excel_dir / filename
        for split, filename in EXCEL_SPLIT_FILES.items()
    }

    for split_name, path in split_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing MUTE {split_name} Excel file: {path}"
            )

    image_index, duplicate_images = find_image_index(raw_root)

    all_rows: List[dict] = []
    stats = {
        "duplicate_image_filenames_ignored": duplicate_images,
        "indexed_image_files": len(image_index),
        "missing_images": 0,
        "invalid_labels": 0,
        "empty_image_names": 0,
        "converted_by_split": {},
    }

    for split_name, file_path in split_files.items():
        print(f"\nProcessing {split_name}: {file_path}")
        rows, missing, invalid, empty_names = read_split(
            split_name=split_name,
            file_path=file_path,
            image_index=image_index,
        )

        print("converted before balancing:", len(rows))
        print("missing images:", missing)
        print("invalid labels:", invalid)
        print("empty image names:", empty_names)

        all_rows.extend(rows)
        stats["missing_images"] += missing
        stats["invalid_labels"] += invalid
        stats["empty_image_names"] += empty_names
        stats["converted_by_split"][split_name] = len(rows)

    return all_rows, stats


# ============================================================
# 4. Balancing
# ============================================================

def balance_rows_combined(
    rows: List[dict],
    seed: int,
    samples_per_class: int | None = None,
) -> List[dict]:
    """
    Balance labels globally after combining all splits.
    """
    rng = random.Random(seed)
    by_label: Dict[int, List[dict]] = defaultdict(list)

    for row in rows:
        by_label[int(row["label"])].append(row)

    missing_labels = sorted(set([0, 1]) - set(by_label))
    if missing_labels:
        raise ValueError(f"Cannot balance: missing labels {missing_labels}")

    target = min(len(by_label[0]), len(by_label[1]))
    if samples_per_class is not None:
        target = int(samples_per_class)

    if target <= 0:
        raise ValueError(f"samples_per_class must be positive, got {target}")

    for label in [0, 1]:
        if len(by_label[label]) < target:
            raise ValueError(
                f"Cannot sample {target} rows for label {label}; "
                f"only {len(by_label[label])} available."
            )

    selected = rng.sample(by_label[0], target) + rng.sample(by_label[1], target)
    rng.shuffle(selected)
    return selected


def balance_rows_per_split(
    rows: List[dict],
    seed: int,
    samples_per_class: int | None = None,
) -> List[dict]:
    """
    Balance labels separately inside each original split.
    """
    rng = random.Random(seed)
    by_split_label: Dict[str, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        by_split_label[str(row["original_split"])][int(row["label"])].append(row)

    selected: List[dict] = []

    for split_name in sorted(by_split_label.keys()):
        label_groups = by_split_label[split_name]

        if 0 not in label_groups or 1 not in label_groups:
            print(
                f"[Warning] Split {split_name!r} does not contain both labels; "
                "skipping this split."
            )
            continue

        target = min(len(label_groups[0]), len(label_groups[1]))
        if samples_per_class is not None:
            target = int(samples_per_class)

        if target <= 0:
            raise ValueError(f"samples_per_class must be positive, got {target}")

        for label in [0, 1]:
            if len(label_groups[label]) < target:
                raise ValueError(
                    f"Cannot sample {target} rows for label {label} in split {split_name}; "
                    f"only {len(label_groups[label])} available."
                )

        split_selected = rng.sample(label_groups[0], target) + rng.sample(label_groups[1], target)
        rng.shuffle(split_selected)
        selected.extend(split_selected)

    rng.shuffle(selected)
    return selected


def select_rows_by_balance_scope(
    rows: List[dict],
    balance_scope: str,
    seed: int,
    samples_per_class: int | None,
) -> List[dict]:
    if balance_scope == "combined":
        return balance_rows_combined(rows, seed=seed, samples_per_class=samples_per_class)
    if balance_scope == "per-split":
        return balance_rows_per_split(rows, seed=seed, samples_per_class=samples_per_class)
    if balance_scope == "none":
        return list(rows)
    raise ValueError("balance_scope must be one of: combined, per-split, none")


# ============================================================
# 5. Standardized output
# ============================================================

def assign_ids_and_copy_images(
    rows: List[dict],
    output_root: Path,
    prefix: str,
) -> List[dict]:
    """
    Copy selected images and assign final globally unique IDs.
    """
    output_image_dir = output_root / "images"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    final_rows: List[dict] = []
    prefix = safe_source_name(prefix)

    for new_i, row in enumerate(rows):
        src_image = Path(row["_src_image_abs"])
        suffix = src_image.suffix.lower()

        row_id = f"{prefix}_{new_i:06d}"
        out_name = f"{row_id}{suffix}"
        out_rel_path = f"images/{out_name}"
        out_abs_path = output_image_dir / out_name

        shutil.copy2(src_image, out_abs_path)

        final_rows.append({
            "id": row_id,
            "image_path": out_rel_path,
            "text": row["text"],
            "label": int(row["label"]),
            "source": "mute",
            "original_split": row["original_split"],
            "original_image_name": row["original_image_name"],
            "original_row_index": int(row["_source_row_index"]),
        })

    return final_rows


def strip_internal_keys(rows: List[dict]) -> List[dict]:
    return [
        {k: v for k, v in row.items() if not k.startswith("_")}
        for row in rows
    ]


def print_distribution(title: str, rows: List[dict]) -> None:
    print(f"\n================ {title} ================")
    print("Rows:", len(rows))
    print("Label counts:", dict(Counter(int(r["label"]) for r in rows)))
    print("Original split counts:", dict(Counter(str(r.get("original_split", "unknown")) for r in rows)))

    split_label_counts: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        split_label_counts[str(row.get("original_split", "unknown"))][int(row["label"])] += 1

    print("Original split + label counts:")
    for split_name in sorted(split_label_counts.keys()):
        print(f"  {split_name}: {dict(split_label_counts[split_name])}")


def save_outputs(output_root: Path, final_rows: List[dict], unbalanced_rows: List[dict]) -> None:
    """
    Save combined standardized dataset and split-specific metadata files.
    """
    write_jsonl(output_root / "dataset_unbalanced_metadata.jsonl", strip_internal_keys(unbalanced_rows))
    write_jsonl(output_root / "dataset.jsonl", final_rows)

    # These split files are useful for audit/debugging. prepare_dataset.py will
    # still create the actual processed_splits/*.jsonl used by training.
    for split_name in ["train", "val", "test"]:
        split_rows = [row for row in final_rows if row["original_split"] == split_name]
        write_jsonl(output_root / f"{split_name}.jsonl", split_rows)


# ============================================================
# 6. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw-root", type=Path, default=RAW_MUTE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DATASET_ROOT)
    parser.add_argument(
        "--balance-scope",
        choices=["combined", "per-split", "none"],
        default=MUTE_BALANCE_SCOPE,
        help=(
            "combined: balance globally after combining train/val/test. "
            "per-split: balance each original split separately. "
            "none: keep all converted rows unbalanced."
        ),
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help="Optional fixed number of samples per class. Default uses minority-class size.",
    )
    parser.add_argument("--id-prefix", type=str, default=MUTE_ID_PREFIX)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_root before writing. Recommended when rebuilding the standardized dataset.",
    )

    return parser.parse_args()


# ============================================================
# 7. Main
# ============================================================

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    ensure_project_dirs()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)

    if not raw_root.exists():
        raise FileNotFoundError(f"RAW_MUTE_ROOT does not exist: {raw_root}")

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    print("================ build_mute_dataset.py ================")
    print("raw_root:", raw_root)
    print("output_root:", output_root)
    print("balance_scope:", args.balance_scope)
    print("seed:", args.seed)
    print("samples_per_class:", args.samples_per_class)
    print("id_prefix:", args.id_prefix)

    all_rows, stats = load_all_raw_rows(raw_root)

    print("\nImage files indexed:", stats["indexed_image_files"])
    print("Duplicate image filenames ignored:", stats["duplicate_image_filenames_ignored"])

    if not all_rows:
        raise ValueError("No valid MUTE rows were converted. Check Excel files, labels, and image paths.")

    print_distribution("Unbalanced converted rows", all_rows)

    selected_rows = select_rows_by_balance_scope(
        rows=all_rows,
        balance_scope=args.balance_scope,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
    )

    if not selected_rows:
        raise ValueError("No rows selected after balancing.")

    print_distribution("Selected rows after balancing", selected_rows)

    final_rows = assign_ids_and_copy_images(
        rows=selected_rows,
        output_root=output_root,
        prefix=args.id_prefix,
    )

    save_outputs(
        output_root=output_root,
        final_rows=final_rows,
        unbalanced_rows=all_rows,
    )

    final_counts = Counter(int(row["label"]) for row in final_rows)

    print("\n================ Summary ================")
    print("Total converted rows before balancing:", len(all_rows))
    print("Total selected rows after balancing:", len(final_rows))
    print("Total missing images:", stats["missing_images"])
    print("Total invalid labels:", stats["invalid_labels"])
    print("Total empty image names:", stats["empty_image_names"])
    print("Final label distribution:", dict(final_counts))
    print("Saved combined:", output_root / "dataset.jsonl")
    print("Saved split files:", output_root / "train.jsonl", output_root / "val.jsonl", output_root / "test.jsonl")
    print("Saved unbalanced metadata:", output_root / "dataset_unbalanced_metadata.jsonl")
    print("Images folder:", output_root / "images")

    if args.balance_scope != "none":
        assert final_counts[0] == final_counts[1], f"Dataset is not balanced: {final_counts}"
        print("Balance check: OK")

    print("\n[Done] Standardized MUTE dataset built successfully.")


if __name__ == "__main__":
    main()
