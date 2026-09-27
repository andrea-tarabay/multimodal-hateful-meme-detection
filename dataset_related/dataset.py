"""
MemeDataset: loads meme_dataset_separated splits for MemeBLIP2 training.

NOTE: This class is NOT used in the current training pipeline.
The current pipeline precomputes all inputs with precompute_inputs.py and
loads them via dataset_loader.py (CachedMemeDataset). This file is the
original on-the-fly loading implementation that predates the caching step
and the switch to CLIP text embeddings.
"""

import json
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

# All available dataset subdirectories
ALL_DATASETS = ["fhm", "harm-c", "harm-p", "mami", "multioff"]


class MemeDataset(Dataset):
    """Binary harmful-meme classification dataset.

    Reads JSONL files from the meme_dataset_separated directory tree:
        <data_root>/<dataset_name>/<split>.jsonl
        <data_root>/<dataset_name>/<image_path>   (e.g. images/fhm_123.png)

    Each JSONL line must have: id, image_path, text, label (0/1), source.
    Missing images are silently skipped.

    Args:
        data_root:        Path to the meme_dataset_separated folder.
        datasets:         List of dataset names to include (subset of ALL_DATASETS).
        split:            "train" | "val" | "test".
        image_processor:  A callable that accepts a PIL Image and returns a dict
                          with key "pixel_values" (e.g. BlipImageProcessor).
        tokenizer:        A BERT-compatible tokenizer for the Q-Former text branch.
        max_text_length:  Maximum token length for text inputs.
    """

    def __init__(
        self,
        data_root: str | Path,
        datasets: List[str],
        split: str,
        image_processor,
        tokenizer,
        max_text_length: int = 64,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

        self.samples: List[dict] = []
        data_root = Path(data_root)

        for ds_name in datasets:
            ds_dir = data_root / ds_name
            jsonl = ds_dir / f"{split}.jsonl"
            if not jsonl.exists():
                continue

            with open(jsonl, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    img_path = ds_dir / item["image_path"]
                    if img_path.exists():
                        self.samples.append({
                            "image_path": str(img_path),
                            "text": item.get("text") or "",
                            "label": int(item["label"]),
                        })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        # Image preprocessing → pixel_values (3, 224, 224)
        image = Image.open(s["image_path"]).convert("RGB")
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Text tokenization (BERT-style for Q-Former)
        enc = self.tokenizer(
            s["text"],
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(s["label"], dtype=torch.long),
        }


def build_datasets(
    data_root: str,
    datasets: List[str],
    image_processor,
    tokenizer,
    max_text_length: int = 64,
) -> dict:
    """Convenience: return a dict of {split: MemeDataset} for train / val / test."""
    splits = {}
    for split in ("train", "val", "test"):
        ds = MemeDataset(
            data_root=data_root,
            datasets=datasets,
            split=split,
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_text_length=max_text_length,
        )
        if len(ds) > 0:
            splits[split] = ds
    return splits
