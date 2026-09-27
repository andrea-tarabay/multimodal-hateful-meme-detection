"""
precompute_inputs.py

Purpose:
    Run input_and_context.py once for train/val/test rows and save model-ready
    tensors.
    
Inputs:
    processed_splits/train_rows.jsonl
    processed_splits/val_rows.jsonl
    processed_splits/test_rows.jsonl

Uses:
    input_and_context.py
        process_dataset_row(row)

Outputs:
    cached_inputs/train/<id>.pt
    cached_inputs/val/<id>.pt
    cached_inputs/test/<id>.pt

Each .pt file contains:
    id
    image_path
    text
    source
    pixel_values
    input_ids
    attention_mask
    clip_text_embedding
    knowledge_embeddings
    knowledge_mask
    labels

Important:
    This script is resumable. If a cached .pt file already exists, it is skipped
    unless FORCE_RECOMPUTE=True.

VERIFYING mode:
    Controlled from config.py.
    If VERIFYING=True, only a few rows are processed and detailed tensor checks
    are printed.
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from config import (
    PROJECT_ROOT,
    DATASET_ROOT,
    CACHE_DIR,
    TRAIN_ROWS_PATH,
    VAL_ROWS_PATH,
    TEST_ROWS_PATH,
    VERIFYING,
    VERIFYING_NUM_ROWS,
    FORCE_RECOMPUTE,
    KNOWLEDGE_DIM,
    KNOWLEDGE_TOP_K,
    CLIP_TEXT_DIM,
    ensure_project_dirs,
)


# ============================================================
# 1. Import preprocessing script
# ============================================================

# Assumption:
#   input_and_context.py is located inside PROJECT_ROOT.
# If it is somewhere else, update PROJECT_ROOT in config.py or move the script.
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import input_and_context
except ImportError as error:
    raise ImportError(
        f"Could not import input_and_context.py from {PROJECT_ROOT}.\n"
        "Make sure input_and_context.py is located in DL_project."
    ) from error

# Very important:
# input_and_context.py joins DATA_ROOT / row["image_path"].
# On the cluster, DATA_ROOT must point to meme_dataset_separated.
input_and_context.DATA_ROOT = DATASET_ROOT


# ============================================================
# 2. JSONL helpers
# ============================================================

def read_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def safe_file_id(row_id: str) -> str:
    """
    Convert a row id into a safe filename.
    """
    return str(row_id).replace("/", "_").replace("\\", "_")


def get_split_path(split_name: str) -> Path:
    if split_name == "train":
        return TRAIN_ROWS_PATH
    if split_name == "val":
        return VAL_ROWS_PATH
    if split_name == "test":
        return TEST_ROWS_PATH
    raise ValueError(f"Unknown split name: {split_name}")


# ============================================================
# 3. Verification helpers
# ============================================================

def print_cached_sample_checks(row: dict, processed: dict, output_path: Path):
    print("\n================ Verification sample ================")
    print("id:", row["id"])
    print("image_path:", row["image_path"])
    print("full image path:", DATASET_ROOT / row["image_path"])
    print("text:", row["text"])
    print("label:", row["label"])
    print("source:", row["source"])
    print("output_path:", output_path)

    print("\nTensor checks:")
    print("pixel_values:", processed["pixel_values"].shape, processed["pixel_values"].dtype)
    print("input_ids:", processed["input_ids"].shape, processed["input_ids"].dtype)
    print("attention_mask:", processed["attention_mask"].shape, processed["attention_mask"].dtype)
    print("clip_text_embedding:", processed["clip_text_embedding"].shape, processed["clip_text_embedding"].dtype)
    print("knowledge_embeddings:", processed["knowledge_embeddings"].shape, processed["knowledge_embeddings"].dtype)
    print("knowledge_mask:", processed["knowledge_mask"].shape, processed["knowledge_mask"].dtype, processed["knowledge_mask"])
    print("labels:", processed["labels"].shape, processed["labels"].dtype, processed["labels"])

    print("\nExpected one-sample shapes:")
    print("pixel_values:         [3, H, W]")
    print("input_ids:            [MAX_TEXT_LENGTH]")
    print("attention_mask:       [MAX_TEXT_LENGTH]")
    print(f"clip_text_embedding:  [{CLIP_TEXT_DIM}]")
    print(f"knowledge_embeddings: [{KNOWLEDGE_TOP_K}, {KNOWLEDGE_DIM}]")
    print(f"knowledge_mask:       [{KNOWLEDGE_TOP_K}]")
    print("labels:               scalar")


def validate_processed_sample(processed: dict):
    """
    Fail early if the cached sample is not compatible with train.py.
    """
    required_keys = [
        "pixel_values",
        "input_ids",
        "attention_mask",
        "clip_text_embedding",
        "knowledge_embeddings",
        "knowledge_mask",
        "labels",
    ]

    missing = [key for key in required_keys if key not in processed]
    if missing:
        raise ValueError(f"Processed sample missing keys: {missing}")

    for key in required_keys:
        if not torch.is_tensor(processed[key]):
            raise TypeError(f"{key} must be a torch.Tensor")

    if processed["pixel_values"].dim() != 3:
        raise ValueError(f"pixel_values should have shape [3, H, W], got {processed['pixel_values'].shape}")

    if processed["input_ids"].dim() != 1:
        raise ValueError(f"input_ids should have shape [L], got {processed['input_ids'].shape}")

    if processed["attention_mask"].dim() != 1:
        raise ValueError(f"attention_mask should have shape [L], got {processed['attention_mask'].shape}")

    if processed["clip_text_embedding"].dim() != 1:
        raise ValueError(
            f"clip_text_embedding should have shape [D], got {processed['clip_text_embedding'].shape}"
        )

    if processed["clip_text_embedding"].shape[0] != CLIP_TEXT_DIM:
        raise ValueError(
            f"clip_text_embedding dim should be {CLIP_TEXT_DIM}, "
            f"got {processed['clip_text_embedding'].shape[0]}"
        )

    if processed["knowledge_embeddings"].dim() != 2:
        raise ValueError(
            f"knowledge_embeddings should have shape [K, D], got {processed['knowledge_embeddings'].shape}"
        )

    if processed["knowledge_mask"].dim() != 1:
        raise ValueError(f"knowledge_mask should have shape [K], got {processed['knowledge_mask'].shape}")

    if processed["labels"].dim() != 0:
        raise ValueError(f"labels should be scalar, got {processed['labels'].shape}")

    label_value = int(processed["labels"].item())
    if label_value not in [0, 1]:
        raise ValueError(f"labels should be 0 or 1, got {label_value}")

    if processed["knowledge_embeddings"].shape[0] != processed["knowledge_mask"].shape[0]:
        raise ValueError("knowledge_embeddings K dimension must match knowledge_mask length")

    if processed["knowledge_embeddings"].shape[1] != KNOWLEDGE_DIM:
        raise ValueError(
            f"knowledge_embeddings dim should be {KNOWLEDGE_DIM}, "
            f"got {processed['knowledge_embeddings'].shape[1]}"
        )

    if processed["knowledge_embeddings"].shape[0] != KNOWLEDGE_TOP_K:
        raise ValueError(
            f"knowledge_embeddings top-k should be {KNOWLEDGE_TOP_K}, "
            f"got {processed['knowledge_embeddings'].shape[0]}"
        )

    if processed["knowledge_mask"].shape[0] != KNOWLEDGE_TOP_K:
        raise ValueError(
            f"knowledge_mask length should be {KNOWLEDGE_TOP_K}, "
            f"got {processed['knowledge_mask'].shape[0]}"
        )

    if torch.isnan(processed["pixel_values"]).any():
        raise ValueError("pixel_values contains NaNs")

    if torch.isnan(processed["clip_text_embedding"]).any():
        raise ValueError("clip_text_embedding contains NaNs")

    if torch.isnan(processed["knowledge_embeddings"]).any():
        raise ValueError("knowledge_embeddings contains NaNs")


# ============================================================
# 4. Precompute one split
# ============================================================

def precompute_split(split_name: str):
    split_path = get_split_path(split_name)
    output_dir = CACHE_DIR / split_name

    if FORCE_RECOMPUTE and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}\n"
            "Run prepare_dataset.py first."
        )

    rows = read_jsonl(split_path)
    print(f"\n================ Processing split: {split_name} ================")
    print(f"Rows loaded: {len(rows)}")
    print(f"Output dir: {output_dir}")

    if VERIFYING:
        rows = rows[:VERIFYING_NUM_ROWS]
        print(f"[Verification] Using only first {len(rows)} row(s).")

    # Checking .exists() on a network filesystem can be slow; print progress.
    print(f"Scanning existing cached files for {split_name}...", flush=True)
    skipped_count = sum(
        1 for row in rows
        if (output_dir / f"{safe_file_id(row['id'])}.pt").exists() and not FORCE_RECOMPUTE
    )
    print(f"Already cached: {skipped_count} | Pending: {len(rows) - skipped_count}", flush=True)

    processed_count = 0
    failed_count = 0

    for row in tqdm(rows, desc=f"Precomputing {split_name}", file=sys.stdout):
        row_id = row["id"]
        output_path = output_dir / f"{safe_file_id(row_id)}.pt"

        if output_path.exists() and not FORCE_RECOMPUTE:
            continue

        try:
            processed = input_and_context.process_dataset_row(row)
            validate_processed_sample(processed)

            cached_sample = {
                "id": row["id"],
                "image_path": row["image_path"],
                "text": row["text"],
                "source": row["source"],
                "pixel_values": processed["pixel_values"].cpu(),
                "input_ids": processed["input_ids"].cpu(),
                "attention_mask": processed["attention_mask"].cpu(),
                "clip_text_embedding": processed["clip_text_embedding"].cpu(),
                "knowledge_embeddings": processed["knowledge_embeddings"].cpu(),
                "knowledge_mask": processed["knowledge_mask"].cpu(),
                "labels": processed["labels"].cpu(),
            }

            torch.save(cached_sample, output_path)
            processed_count += 1

            if VERIFYING:
                print_cached_sample_checks(row, processed, output_path)

        except Exception as error:
            failed_count += 1
            print(f"\n[Warning] Failed row id={row_id}: {error}")

    print(f"\nSummary for split: {split_name}")
    print("Processed:", processed_count)
    print("Skipped existing:", skipped_count)
    print("Failed:", failed_count)


# ============================================================
# 5. Main
# ============================================================

def main():
    ensure_project_dirs()

    print("================ precompute_inputs.py ================")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATASET_ROOT:", DATASET_ROOT)
    print("CACHE_DIR:", CACHE_DIR)
    print("TRAIN_ROWS_PATH:", TRAIN_ROWS_PATH)
    print("VAL_ROWS_PATH:", VAL_ROWS_PATH)
    print("TEST_ROWS_PATH:", TEST_ROWS_PATH)
    print("VERIFYING:", VERIFYING)
    print("VERIFYING_NUM_ROWS:", VERIFYING_NUM_ROWS)
    print("FORCE_RECOMPUTE:", FORCE_RECOMPUTE)
    print("KNOWLEDGE_DIM:", KNOWLEDGE_DIM)
    print("KNOWLEDGE_TOP_K:", KNOWLEDGE_TOP_K)
    print("CLIP_TEXT_DIM:", CLIP_TEXT_DIM)
    print("input_and_context.DATA_ROOT:", input_and_context.DATA_ROOT)

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(f"PROJECT_ROOT does not exist: {PROJECT_ROOT}")

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"DATASET_ROOT does not exist: {DATASET_ROOT}")

    for split_name in ["train", "val", "test"]:
        precompute_split(split_name)

    print("\n[Done] Input/context tensors cached successfully.")


if __name__ == "__main__":
    main()
