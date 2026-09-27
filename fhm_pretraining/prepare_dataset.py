"""
prepare_dataset.py

Purpose:
    Load the raw meme dataset, verify rows/images, create train/val/test splits,
    and save split metadata as JSONL files.

This script should be run once before preprocessing/training.

Outputs:
    processed_splits/train_rows.jsonl
    processed_splits/val_rows.jsonl
    processed_splits/test_rows.jsonl

VERIFYING mode:
    Controlled from config.py.
    If VERIFYING=True, the script loads only a few rows and prints detailed checks.
"""

import json
import random
from pathlib import Path

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
    ensure_project_dirs,
)

try:
    from config import BALANCE_CLASSES
except ImportError:
    BALANCE_CLASSES = False


# ============================================================
# 1. Helpers
# ============================================================

def load_official_splits(dataset_root: Path):
    """Load the official FHM train/val/test JSONL files from dataset_root."""
    splits = {}
    for name in ("train", "val", "test"):
        path = dataset_root / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Official {name} split not found: {path}\n"
                "Expected train.jsonl, val.jsonl, and test.jsonl in DATASET_ROOT."
            )
        df = pd.read_json(path, lines=True)
        print(f"[Info] Loaded {len(df)} rows from {path}")
        splits[name] = df
    return splits["train"], splits["val"], splits["test"]


def find_dataset_file(dataset_root: Path) -> Path:
    """
    Try to automatically find the main dataset metadata file.

    Supported formats:
        .jsonl
        .json
        .csv
        .parquet

    If several files are found, the first one in sorted order is used.
    """
    candidates = []

    for pattern in ["*.jsonl", "*.json", "*.csv", "*.parquet"]:
        candidates.extend(dataset_root.rglob(pattern))

    candidates = sorted(candidates)

    if not candidates:
        raise FileNotFoundError(
            f"No dataset metadata file found under: {dataset_root}\n"
            "Expected one of: .jsonl, .json, .csv, .parquet"
        )

    print(f"[Info] Found dataset metadata file: {candidates[0]}")
    return candidates[0]


def load_rows(dataset_file: Path) -> pd.DataFrame:
    """
    Load dataset rows into a pandas DataFrame.
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


def check_required_columns(df: pd.DataFrame):
    """
    Ensure the dataset has all fields needed by input_and_context.py.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only required columns, normalize types, and remove invalid rows.
    """
    df = df.copy()

    check_required_columns(df)

    df = df[REQUIRED_COLUMNS].copy()

    df["id"] = df["id"].astype(str)
    df["image_path"] = df["image_path"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["source"] = df["source"].fillna("unknown").astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Keep binary labels only.
    df = df[df["label"].isin([0, 1])].copy()
    after = len(df)

    if before != after:
        print(f"[Warning] Removed {before - after} rows with invalid labels.")

    # Remove duplicate ids if any.
    before = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first").copy()
    after = len(df)

    if before != after:
        print(f"[Warning] Removed {before - after} duplicate id rows.")

    df = df.reset_index(drop=True)
    return df


def image_exists(row, dataset_root: Path) -> bool:
    """
    Check whether image_path exists relative to DATASET_ROOT.
    """
    image_path = dataset_root / row["image_path"]
    return image_path.exists()


def verify_image_paths(df: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    """
    Remove rows whose image files do not exist.
    """
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

    df = df.loc[exists_mask].copy().reset_index(drop=True)
    return df


def save_jsonl(rows, output_path: Path):
    """
    Save a list of dictionaries as JSONL.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_split_summary(train_df, val_df, test_df):
    """
    Print split sizes and label distributions.
    """
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


def print_verification_examples(df: pd.DataFrame, dataset_root: Path, title: str):
    """
    Print detailed row checks when VERIFYING=True.
    """
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


def stratified_split(df: pd.DataFrame):
    """
    Create train/val/test splits.

    Tries stratification by label. If the dataset is too small for stratified
    splitting, falls back to random splitting.
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

def balance_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Undersample majority class so both classes have equal size."""
    hate_df = df[df["label"] == 1]
    non_hate_df = df[df["label"] == 0]
    min_size = min(len(hate_df), len(non_hate_df))
    hate_sampled = hate_df.sample(n=min_size, random_state=seed)
    non_hate_sampled = non_hate_df.sample(n=min_size, random_state=seed)
    return pd.concat([hate_sampled, non_hate_sampled]).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)


def make_verification_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Select a smaller subset for verification/mini-runs.

    Unlike df.head(n), this samples randomly and tries to preserve
    the label distribution.
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
# 2. Main
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
        raise FileNotFoundError(f"DATASET_ROOT does not exist: {DATASET_ROOT}")

    train_df, val_df, test_df = load_official_splits(DATASET_ROOT)

    split_dfs = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n[Info] Normalizing {name} split ({len(df)} rows)...")
        df = normalize_rows(df)
        df = verify_image_paths(df, DATASET_ROOT)
        print(f"[Info] {name} rows after normalization + image check: {len(df)}")

        if BALANCE_CLASSES:
            before = len(df)
            df = balance_split(df, seed=RANDOM_SEED)
            print(f"[Info] {name} rows after class balancing: {before} → {len(df)}")

        if VERIFYING:
            df = make_verification_subset(df=df, n=VERIFYING_NUM_ROWS, seed=RANDOM_SEED)
            print(f"[Verification] {name}: using {len(df)} rows.")

        split_dfs[name] = df

    train_df = split_dfs["train"]
    val_df = split_dfs["val"]
    test_df = split_dfs["test"]

    print_split_summary(train_df, val_df, test_df)

    train_rows = train_df.to_dict(orient="records")
    val_rows = val_df.to_dict(orient="records")
    test_rows = test_df.to_dict(orient="records")

    save_jsonl(train_rows, TRAIN_ROWS_PATH)
    save_jsonl(val_rows, VAL_ROWS_PATH)
    save_jsonl(test_rows, TEST_ROWS_PATH)

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
