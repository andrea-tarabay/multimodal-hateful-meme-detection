"""
dataset_loader.py

Reusable PyTorch Dataset/DataLoader utilities for cached MUTE meme inputs.

This script loads the .pt files created by precompute_inputs.py:
    cached_inputs/train/<id>.pt
    cached_inputs/val/<id>.pt
    cached_inputs/test/<id>.pt

Optionally, it also loads teacher probabilities created by:
    generate_teacher_outputs.py

Used by:
    train_all_modes.py
    evaluate.py

Dataset output without distillation:
    pixel_values
    input_ids
    attention_mask
    clip_text_embedding
    knowledge_embeddings
    knowledge_mask
    labels
    id
    text
    image_path
    source

Dataset output with distillation:
    all of the above + teacher_probs
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    PROJECT_ROOT,
    CACHE_DIR,
    TRAIN_TEACHER_OUTPUT_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    VERIFYING,
)


# ============================================================
# 1. Helpers
# ============================================================

def load_teacher_outputs(path: Path = TRAIN_TEACHER_OUTPUT_PATH):
    """
    Load teacher probabilities by id.

    Supports the updated first-stage format:
        prob_not_hate, prob_hate

    Also tolerates older format:
        teacher_probs = [prob_not_hate, prob_hate]

    Returns:
        dict[id] = torch.FloatTensor([prob_not_hate, prob_hate])
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Teacher output file not found: {path}\n"
            "Run generate_teacher_outputs.py first, or set use_teacher=False."
        )

    teacher_by_id = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            row_id = str(row["id"])

            if "prob_not_hate" in row and "prob_hate" in row:
                prob_not_hate = float(row["prob_not_hate"])
                prob_hate = float(row["prob_hate"])
            elif "teacher_probs" in row:
                probs_raw = row["teacher_probs"]
                prob_not_hate = float(probs_raw[0])
                prob_hate = float(probs_raw[1])
            else:
                continue

            probs = torch.tensor([prob_not_hate, prob_hate], dtype=torch.float32)
            probs = probs / probs.sum().clamp_min(1e-8)
            teacher_by_id[row_id] = probs

    if not teacher_by_id:
        raise ValueError(
            f"No usable teacher probabilities found in {path}. "
            "Expected prob_not_hate/prob_hate or teacher_probs."
        )

    return teacher_by_id


def list_cached_files(cache_dir: Path, split: str):
    """
    List cached .pt files for one split.
    """
    split_dir = Path(cache_dir) / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Cached split directory not found: {split_dir}\n"
            "Run precompute_inputs.py first."
        )

    files = sorted(split_dir.glob("*.pt"))

    if not files:
        raise FileNotFoundError(
            f"No .pt files found in {split_dir}\n"
            "Run precompute_inputs.py first, or check VERIFYING mode."
        )

    return files


def move_batch_to_device(batch, device):
    """
    Move tensor values in a batch to device. Metadata strings/lists stay unchanged.
    """
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


# ============================================================
# 2. Dataset
# ============================================================

class CachedMemeDataset(Dataset):
    """
    Dataset for cached model-ready meme tensors.

    Args:
        split:
            "train", "val", or "test".

        cache_dir:
            Directory containing cached_inputs/{split}/*.pt.

        use_teacher:
            If True, attach teacher_probs to each sample.
            Usually True only for the train split during distillation.

        teacher_output_path:
            JSONL file with teacher probabilities.

        verifying:
            If True, print checks for the first sample.
    """

    def __init__(
        self,
        split: str,
        cache_dir: Path = CACHE_DIR,
        use_teacher: bool = False,
        teacher_output_path: Path = TRAIN_TEACHER_OUTPUT_PATH,
        verifying: bool = VERIFYING,
    ):
        self.split = split
        self.cache_dir = Path(cache_dir)
        self.use_teacher = bool(use_teacher)
        self.teacher_output_path = Path(teacher_output_path)
        self.verifying = bool(verifying)

        self.files = list_cached_files(self.cache_dir, self.split)

        self.teacher_by_id = None
        if self.use_teacher:
            self.teacher_by_id = load_teacher_outputs(self.teacher_output_path)

        if self.verifying:
            print(f"\n[Dataset verification] split={self.split}")
            print("PROJECT_ROOT:", PROJECT_ROOT)
            print("cache_dir:", self.cache_dir)
            print("num files:", len(self.files))
            print("use_teacher:", self.use_teacher)
            if self.use_teacher:
                print("teacher_output_path:", self.teacher_output_path)
                print("teacher outputs:", len(self.teacher_by_id))
            sample = self[0]
            self.print_sample_checks(sample)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        sample = torch.load(path, map_location="cpu", weights_only=True)

        if "clip_text_embedding" not in sample:
            raise KeyError(
                f"Missing clip_text_embedding in cached file: {path}\n"
                "Run the updated precompute_inputs.py to regenerate cached inputs."
            )

        output = {
            "pixel_values": sample["pixel_values"].float(),
            "input_ids": sample["input_ids"].long(),
            "attention_mask": sample["attention_mask"].long(),
            "clip_text_embedding": sample["clip_text_embedding"].float(),
            "knowledge_embeddings": sample["knowledge_embeddings"].float(),
            "knowledge_mask": sample["knowledge_mask"].bool(),
            "labels": sample["labels"].long(),
            "id": str(sample["id"]),
            "text": sample["text"],
            "image_path": sample["image_path"],
            "source": sample["source"],
        }

        # Preserve optional MUTE metadata for analysis/debugging when available.
        for optional_key in ["original_split", "original_image_name", "original_row_index"]:
            if optional_key in sample:
                output[optional_key] = sample[optional_key]

        if self.use_teacher:
            row_id = str(sample["id"])
            if row_id not in self.teacher_by_id:
                raise KeyError(
                    f"No teacher output found for id={row_id}.\n"
                    "Run generate_teacher_outputs.py for the train split."
                )
            output["teacher_probs"] = self.teacher_by_id[row_id]

        return output

    @staticmethod
    def print_sample_checks(sample: dict):
        print("\nFirst sample checks:")
        print("id:", sample["id"])
        print("image_path:", sample["image_path"])
        print("text:", sample["text"])
        print("source:", sample["source"])
        if "original_split" in sample:
            print("original_split:", sample["original_split"])
        print("labels:", sample["labels"], sample["labels"].shape, sample["labels"].dtype)
        print("pixel_values:", sample["pixel_values"].shape, sample["pixel_values"].dtype)
        print("input_ids:", sample["input_ids"].shape, sample["input_ids"].dtype)
        print("attention_mask:", sample["attention_mask"].shape, sample["attention_mask"].dtype)
        print("clip_text_embedding:", sample["clip_text_embedding"].shape, sample["clip_text_embedding"].dtype)
        print("knowledge_embeddings:", sample["knowledge_embeddings"].shape, sample["knowledge_embeddings"].dtype)
        print("knowledge_mask:", sample["knowledge_mask"].shape, sample["knowledge_mask"].dtype, sample["knowledge_mask"])
        if "teacher_probs" in sample:
            print("teacher_probs:", sample["teacher_probs"], sample["teacher_probs"].shape, sample["teacher_probs"].dtype)


# ============================================================
# 3. DataLoader builders
# ============================================================

def build_dataloader(
    split: str,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = None,
    num_workers: int = NUM_WORKERS,
    cache_dir: Path = CACHE_DIR,
    use_teacher: bool = False,
    teacher_output_path: Path = TRAIN_TEACHER_OUTPUT_PATH,
    verifying: bool = VERIFYING,
):
    """
    Build one DataLoader.
    """
    if shuffle is None:
        shuffle = split == "train"

    dataset = CachedMemeDataset(
        split=split,
        cache_dir=cache_dir,
        use_teacher=use_teacher,
        teacher_output_path=teacher_output_path,
        verifying=verifying,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


def build_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    cache_dir: Path = CACHE_DIR,
    use_teacher_for_train: bool = False,
    teacher_output_path: Path = TRAIN_TEACHER_OUTPUT_PATH,
    verifying: bool = VERIFYING,
):
    """
    Build train/val/test DataLoaders.

    Teacher probabilities are normally attached only to the train loader.
    """
    train_loader = build_dataloader(
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        cache_dir=cache_dir,
        use_teacher=use_teacher_for_train,
        teacher_output_path=teacher_output_path,
        verifying=verifying,
    )

    val_loader = build_dataloader(
        split="val",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        cache_dir=cache_dir,
        use_teacher=False,
        teacher_output_path=teacher_output_path,
        verifying=verifying,
    )

    test_loader = build_dataloader(
        split="test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        cache_dir=cache_dir,
        use_teacher=False,
        teacher_output_path=teacher_output_path,
        verifying=verifying,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# 4. Standalone verification
# ============================================================

def main():
    print("================ dataset_loader.py ================")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CACHE_DIR:", CACHE_DIR)
    print("TRAIN_TEACHER_OUTPUT_PATH:", TRAIN_TEACHER_OUTPUT_PATH)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("NUM_WORKERS:", NUM_WORKERS)
    print("VERIFYING:", VERIFYING)

    loader = build_dataloader(
        split="train",
        batch_size=min(2, BATCH_SIZE),
        shuffle=False,
        num_workers=0,
        use_teacher=False,
        verifying=True,
    )

    batch = next(iter(loader))

    print("\n================ Batch checks ================")
    print("pixel_values:", batch["pixel_values"].shape, batch["pixel_values"].dtype)
    print("input_ids:", batch["input_ids"].shape, batch["input_ids"].dtype)
    print("attention_mask:", batch["attention_mask"].shape, batch["attention_mask"].dtype)
    print("clip_text_embedding:", batch["clip_text_embedding"].shape, batch["clip_text_embedding"].dtype)
    print("knowledge_embeddings:", batch["knowledge_embeddings"].shape, batch["knowledge_embeddings"].dtype)
    print("knowledge_mask:", batch["knowledge_mask"].shape, batch["knowledge_mask"].dtype)
    print("labels:", batch["labels"].shape, batch["labels"].dtype)
    print("ids:", batch["id"])

    print("\nExpected batch shapes:")
    print("pixel_values:         [B, 3, H, W]")
    print("input_ids:            [B, 128]")
    print("attention_mask:       [B, 128]")
    print("clip_text_embedding:  [B, 512]")
    print("knowledge_embeddings: [B, 5, 384]")
    print("knowledge_mask:       [B, 5]")
    print("labels:               [B]")

    print("\n[Done] Dataset/DataLoader verification successful.")


if __name__ == "__main__":
    main()
