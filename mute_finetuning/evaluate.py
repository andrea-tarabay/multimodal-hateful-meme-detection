"""
evaluate.py

Evaluate a trained MUTE fine-tuned checkpoint on the cached test split.

Inputs:
    cached_inputs/test/*.pt
    checkpoints/{mode}_best.pt or checkpoints/{mode}_last.pt

Outputs:
    results/test_metrics_{mode}_{checkpoint_type}.json
    results/test_predictions_{mode}_{checkpoint_type}.csv
    plots/confusion_matrix_{mode}_{checkpoint_type}.png

Supported modes:
    memeblip2_only
    context_only
    context_distillation
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    NUM_WORKERS,
    PLOT_DIR,
    RESULTS_DIR,
    VALID_TRAIN_MODES,
    VERIFYING,
    ensure_project_dirs,
)
from dataset_loader import build_dataloaders, move_batch_to_device
from knowledge_gated import CrossAttentionKnowledgeMemeBLIP2
from memeblip2 import MemeBLIP2


# ============================================================
# 1. Model construction
# ============================================================

def build_model(mode: str, device: torch.device):
    if mode == "memeblip2_only":
        model = MemeBLIP2()
    elif mode in {"context_only", "context_distillation"}:
        model = CrossAttentionKnowledgeMemeBLIP2()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return model.to(device)


# ============================================================
# 2. Checkpoint loading
# ============================================================

def extract_state_dict(checkpoint):
    """
    Accept both full training checkpoints and raw state_dict files.
    """
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint
    raise ValueError("Unsupported checkpoint format.")


def load_checkpoint(model, checkpoint_path: str | torch.PathLike, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint


# ============================================================
# 3. Forward pass
# ============================================================

def forward_batch(model, mode: str, batch: dict):
    if mode == "memeblip2_only":
        return model(
            pixel_values=batch["pixel_values"],
            clip_text_embedding=batch["clip_text_embedding"],
        )

    return model(
        pixel_values=batch["pixel_values"],
        clip_text_embedding=batch["clip_text_embedding"],
        knowledge_embeddings=batch["knowledge_embeddings"],
        knowledge_mask=batch["knowledge_mask"],
    )


# ============================================================
# 4. Metrics / plots
# ============================================================

def compute_metrics(labels, preds, probs):
    accuracy = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = None

    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": None if auc is None else float(auc),
        "confusion_matrix": cm,
    }


def save_confusion_matrix(labels, preds, output_path):
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["not-hateful", "hateful"],
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    display.plot(ax=ax, values_format="d")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


# ============================================================
# 5. Evaluation
# ============================================================

def evaluate(mode: str, checkpoint_type: str, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
    ensure_project_dirs()

    if mode not in VALID_TRAIN_MODES:
        raise ValueError(f"Invalid mode={mode}. Valid modes: {VALID_TRAIN_MODES}")

    if checkpoint_type not in {"best", "last"}:
        raise ValueError("checkpoint_type must be 'best' or 'last'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = CHECKPOINT_DIR / f"{mode}_{checkpoint_type}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("==============================================")
    print(f"Evaluating mode: {mode}")
    print(f"Checkpoint type: {checkpoint_type}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print("==============================================")

    _, _, test_loader = build_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        use_teacher_for_train=False,
        verifying=VERIFYING,
    )

    model = build_model(mode, device)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    all_ids = []
    all_image_paths = []
    all_texts = []
    all_sources = []
    all_original_splits = []
    all_labels = []
    all_preds = []
    all_prob_hateful = []
    all_prob_not_hateful = []

    with torch.no_grad():
        for batch in test_loader:
            batch = move_batch_to_device(batch, device)
            logits = forward_batch(model, mode, batch)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_ids.extend(batch["id"])
            all_image_paths.extend(batch["image_path"])
            all_texts.extend(batch["text"])
            all_sources.extend(batch["source"])
            all_labels.extend(batch["labels"].detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())
            all_prob_not_hateful.extend(probs[:, 0].detach().cpu().tolist())
            all_prob_hateful.extend(probs[:, 1].detach().cpu().tolist())

            if "original_split" in batch:
                all_original_splits.extend(batch["original_split"])
            else:
                all_original_splits.extend([None] * len(batch["id"]))

    metrics = compute_metrics(
        labels=all_labels,
        preds=all_preds,
        probs=all_prob_hateful,
    )

    metrics["mode"] = mode
    metrics["checkpoint_type"] = checkpoint_type
    metrics["checkpoint_path"] = str(checkpoint_path)
    metrics["num_examples"] = len(all_labels)

    if isinstance(checkpoint, dict):
        for key in ["epoch", "best_val_metric", "best_val_f1", "best_val_acc", "val_metrics"]:
            if key in checkpoint:
                try:
                    json.dumps(checkpoint[key])
                    metrics[f"checkpoint_{key}"] = checkpoint[key]
                except TypeError:
                    metrics[f"checkpoint_{key}"] = str(checkpoint[key])

    metrics_path = RESULTS_DIR / f"test_metrics_{mode}_{checkpoint_type}.json"
    predictions_path = RESULTS_DIR / f"test_predictions_{mode}_{checkpoint_type}.csv"
    confusion_matrix_path = PLOT_DIR / f"confusion_matrix_{mode}_{checkpoint_type}.png"

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    predictions_df = pd.DataFrame({
        "id": all_ids,
        "image_path": all_image_paths,
        "text": all_texts,
        "source": all_sources,
        "original_split": all_original_splits,
        "label": all_labels,
        "pred": all_preds,
        "prob_not_hateful": all_prob_not_hateful,
        "prob_hateful": all_prob_hateful,
    })

    predictions_df.to_csv(predictions_path, index=False)

    save_confusion_matrix(
        labels=all_labels,
        preds=all_preds,
        output_path=confusion_matrix_path,
    )

    print("\n================ Test metrics ================")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n================ Saved files ================")
    print(metrics_path)
    print(predictions_path)
    print(confusion_matrix_path)


# ============================================================
# 6. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=VALID_TRAIN_MODES,
        help="Training mode to evaluate.",
    )
    parser.add_argument(
        "--checkpoint-type",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Which checkpoint to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--verifying",
        action="store_true",
        help="Accepted for compatibility with run scripts; actual VERIFYING is controlled in config.py.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    evaluate(
        mode=args.mode,
        checkpoint_type=args.checkpoint_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
