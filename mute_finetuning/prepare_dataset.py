"""
prepare_dataset.py

Load the standardized MUTE dataset created by build_mute_dataset.py, verify the
rows/images, create train/val/test metadata files, and save them as JSONL.

Inputs:
    DATASET_ROOT/dataset.jsonl

Outputs:
    processed_splits/train_rows.jsonl
    processed_splits/val_rows.jsonl
    processed_splits/test_rows.jsonl

Expected standardized row format:
    {
        "id": "mute_000001",
        "image_path": "images/mute_000001.jpg",
        "text": "...",
        "label": 0 or 1,
        "source": "mute",
        ... optional metadata ...
    }

VERIFYING mode:
    Controlled from config.py. If VERIFYING=True, a stratified subset is sampled
    before splitting.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATASET_ROOT,
    PROJECT_ROOT,
    SPLIT_DIR,
    TRAIN_ROWS_PATH,
    VAL_ROWS_PATH,
    TEST_ROWS_PATH,
    VERIFYING,
    VERIFYING_NUM_ROWS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    REQUIRED_COLUMNS,
    BALANCE_CLASSES,
    ensure_project_dirs,
)


# ============================================================
# 1. File loading
# ============================================================

def get_standard_dataset_path(dataset_root: Path = DATASET_ROOT) -> Path:
    """
    Prefer DATASET_ROOT/dataset.jsonl, which is produced by build_mute_dataset.py.
    """
    path = Path(dataset_root) / "dataset.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Standardized dataset not found: {path}\n"
            "Run build_mute_dataset.py first."
        )
    return path


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rows(dataset_file: Path) -> pd.DataFrame:
    """
    Load standardized MUTE rows into a DataFrame.
    """
    suffix = dataset_file.suffix.lower()

    if suffix == ".jsonl":
        df = pd.read_json(dataset_file, lines=True)
    elif suffix == ".json":
        df = pd.read_json(dataset_file)
    elif suffix == ".csv":
        df = pd.read_csv(dataset_file)
    elif suffix == ".parquet":
        df = pd.read_parquet(dataset_file)
    else:
        raise ValueError(f"Unsupported dataset file format: {dataset_file}")

    return df


# ============================================================
# 2. Row validation / normalization
# ============================================================

def check_required_columns(df: pd.DataFrame):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the standardized dataset while preserving useful optional metadata.
    """
    df = df.copy()
    check_required_columns(df)

    df["id"] = df["id"].astype(str)
    df["image_path"] = df["image_path"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["source"] = df["source"].fillna("mute").astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])].copy()
    after = len(df)

    if before != after:
        print(f"[Warning] Removed {before - after} rows with invalid labels.")

    before = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first").copy()
    after = len(df)

    if before != after:
        print(f"[Warning] Removed {before - after} duplicate id rows.")

    df = df.reset_index(drop=True)
    return df


def image_exists(row, dataset_root: Path) -> bool:
    return (dataset_root / row["image_path"]).exists()


def verify_image_paths(df: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    exists_mask = df.apply(lambda row: image_exists(row, dataset_root), axis=1)
    missing_count = int((~exists_mask).sum())

    if missing_count > 0:
        print(f"[Warning] Found {missing_count} rows with missing images. Removing them.")

        if VERIFYING:
            missing_examples = df.loc[~exists_mask].head(10)
            print("\n[Verification] Missing image examples:")
            for _, row in missing_examples.iterrows():
                print("  id:", row["id"])
                print("  expected:", dataset_root / row["image_path"])

    return df.loc[exists_mask].copy().reset_index(drop=True)


# ============================================================
# 3. Optional balancing / verification subset
# ============================================================

def balance_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Optional undersampling. Usually not needed for MUTE if build_mute_dataset.py
    already used MUTE_BALANCE_SCOPE='combined'.
    """
    hate_df = df[df["label"] == 1]
    non_hate_df = df[df["label"] == 0]

    min_size = min(len(hate_df), len(non_hate_df))
    if min_size == 0:
        raise ValueError("Cannot balance because one label class is empty.")

    hate_sampled = hate_df.sample(n=min_size, random_state=seed)
    non_hate_sampled = non_hate_df.sample(n=min_size, random_state=seed)

    return pd.concat([hate_sampled, non_hate_sampled]).sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)


def make_verification_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Select a smaller randomized subset while preserving label distribution when possible.
    """
    if n is None or n <= 0 or n >= len(df):
        return df.copy().reset_index(drop=True)

    stratify_labels = None
    if df["label"].nunique() == 2 and df["label"].value_counts().min() >= 2:
        stratify_labels = df["label"]

    try:
        subset_df, _ = train_test_split(
            df,
            train_size=n,
            random_state=seed,
            shuffle=True,
            stratify=stratify_labels,
        )
    except ValueError as error:
        print(f"[Warning] Stratified verification subset failed: {error}")
        print("[Warning] Falling back to random subset.")
        subset_df = df.sample(n=n, random_state=seed)

    return subset_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ============================================================
# 4. Splitting
# ============================================================

def use_original_splits_if_available(df: pd.DataFrame) -> bool:
    """
    Use original MUTE split assignments when all three are present.

    This avoids doing two independent splitting stages after build_mute_dataset.py
    already preserved train/val/test membership.
    """
    if "original_split" not in df.columns:
        return False

    split_values = set(df["original_split"].dropna().astype(str).str.lower().tolist())
    return {"train", "val", "test"}.issubset(split_values)


def split_by_original_split(df: pd.DataFrame):
    split_col = df["original_split"].astype(str).str.lower()

    train_df = df[split_col == "train"].copy().reset_index(drop=True)
    val_df = df[split_col == "val"].copy().reset_index(drop=True)
    test_df = df[split_col == "test"].copy().reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "original_split exists, but at least one split is empty: "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    return train_df, val_df, test_df


def stratified_split(df: pd.DataFrame):
    """
    Create train/val/test splits using configured ratios.
    Used only when original split metadata is unavailable or intentionally ignored.
    """
    if not abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-8:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")

    stratify_labels = df["label"] if df["label"].nunique() == 2 else None

    try:
        train_df, temp_df = train_test_split(
            df,
            train_size=TRAIN_RATIO,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=stratify_labels,
        )

        relative_val_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        temp_stratify = temp_df["label"] if temp_df["label"].nunique() == 2 else None

        val_df, test_df = train_test_split(
            temp_df,
            train_size=relative_val_ratio,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=temp_stratify,
        )

    except ValueError as error:
        print(f"[Warning] Stratified split failed: {error}")
        print("[Warning] Falling back to non-stratified random split.")

        train_df, temp_df = train_test_split(
            df,
            train_size=TRAIN_RATIO,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=None,
        )

        relative_val_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=relative_val_ratio,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=None,
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ============================================================
# 5. Saving / reporting
# ============================================================

def dataframe_to_jsonable_rows(df: pd.DataFrame) -> List[dict]:
    """
    Convert pandas rows to JSON-safe dictionaries.
    Keeps optional metadata columns when present.
    """
    df = df.copy()
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")


def save_jsonl(rows: Iterable[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_split_summary(train_df, val_df, test_df):
    print("\n================ Split summary ================")
    print(f"Train size: {len(train_df)}")
    print(f"Val size:   {len(val_df)}")
    print(f"Test size:  {len(test_df)}")

    for name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        print(f"\n{name} label distribution:")
        print(split_df["label"].value_counts(normalize=False).sort_index())
        print(f"{name} source distribution:")
        print(split_df["source"].value_counts(normalize=False).head(10))

        if "original_split" in split_df.columns:
            print(f"{name} original_split distribution:")
            print(split_df["original_split"].value_counts(normalize=False).head(10))


def print_verification_examples(df: pd.DataFrame, dataset_root: Path, title: str):
    print(f"\n================ {title} ================")

    for idx, row in df.iterrows():
        full_image_path = dataset_root / row["image_path"]
        print(f"\nRow {idx}")
        print("id:        ", row["id"])
        print("image_path:", row["image_path"])
        print("full path: ", full_image_path)
        print("exists:    ", full_image_path.exists())
        print("text:      ", row["text"])
        print("label:     ", row["label"])
        print("source:    ", row["source"])
        if "original_split" in row:
            print("orig split:", row["original_split"])


# ============================================================
# 6. Main
# ============================================================

def main():
    random.seed(RANDOM_SEED)
    ensure_project_dirs()

    print("================ prepare_dataset.py ================")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATASET_ROOT:", DATASET_ROOT)
    print("SPLIT_DIR:", SPLIT_DIR)
    print("TRAIN_ROWS_PATH:", TRAIN_ROWS_PATH)
    print("VAL_ROWS_PATH:", VAL_ROWS_PATH)
    print("TEST_ROWS_PATH:", TEST_ROWS_PATH)
    print("VERIFYING:", VERIFYING)
    print("VERIFYING_NUM_ROWS:", VERIFYING_NUM_ROWS)
    print("BALANCE_CLASSES:", BALANCE_CLASSES)
    print("TRAIN_RATIO:", TRAIN_RATIO)
    print("VAL_RATIO:", VAL_RATIO)
    print("TEST_RATIO:", TEST_RATIO)
    print("RANDOM_SEED:", RANDOM_SEED)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"DATASET_ROOT does not exist: {DATASET_ROOT}\n"
            "Run build_mute_dataset.py first."
        )

    dataset_file = get_standard_dataset_path(DATASET_ROOT)
    df = load_rows(dataset_file)

    print(f"\n[Info] Loaded {len(df)} rows from {dataset_file}")
    print("[Info] Columns:", list(df.columns))

    df = normalize_rows(df)
    print(f"[Info] Rows after normalization: {len(df)}")

    df = verify_image_paths(df, DATASET_ROOT)
    print(f"[Info] Rows after image-path verification: {len(df)}")

    if BALANCE_CLASSES:
        before = len(df)
        df = balance_split(df, seed=RANDOM_SEED)
        print(f"[Info] Rows after class balancing: {before} -> {len(df)}")

    if VERIFYING:
        before_subset = len(df)
        df = make_verification_subset(
            df=df,
            n=VERIFYING_NUM_ROWS,
            seed=RANDOM_SEED,
        )
        print(
            f"\n[Verification] Using {len(df)} randomly sampled/stratified rows "
            f"from {before_subset} total rows."
        )
        print_verification_examples(
            df.head(min(20, len(df))),
            DATASET_ROOT,
            "Loaded subset examples before split",
        )

    if len(df) < 3:
        raise ValueError(
            f"Need at least 3 rows to create train/val/test split, got {len(df)}. "
            "Increase VERIFYING_NUM_ROWS in config.py or set VERIFYING=False."
        )

    if use_original_splits_if_available(df):
        print("\n[Info] Using original MUTE train/val/test split assignments.")
        train_df, val_df, test_df = split_by_original_split(df)
    else:
        print("\n[Info] original_split missing/incomplete. Creating stratified random splits.")
        train_df, val_df, test_df = stratified_split(df)

    print_split_summary(train_df, val_df, test_df)

    save_jsonl(dataframe_to_jsonable_rows(train_df), TRAIN_ROWS_PATH)
    save_jsonl(dataframe_to_jsonable_rows(val_df), VAL_ROWS_PATH)
    save_jsonl(dataframe_to_jsonable_rows(test_df), TEST_ROWS_PATH)

    print("\n================ Saved files ================")
    print(TRAIN_ROWS_PATH)
    print(VAL_ROWS_PATH)
    print(TEST_ROWS_PATH)

    if VERIFYING:
        print_verification_examples(train_df.head(20), DATASET_ROOT, "Train split examples")
        print_verification_examples(val_df.head(20), DATASET_ROOT, "Validation split examples")
        print_verification_examples(test_df.head(20), DATASET_ROOT, "Test split examples")

    print("\n[Done] Dataset splits prepared successfully.")


if __name__ == "__main__":
    main()
