"""
evaluate.py

Purpose:
    Evaluate a saved checkpoint on the cached test split.

Supported modes:
    1. memeblip2_only
    2. context_only
    3. context_distillation

Inputs:
    cached_inputs/test/*.pt
    checkpoints/<mode>_best.pt or checkpoints/<mode>_last.pt

Cached inputs must include precomputed clip_text_embedding.

Outputs:
    results/test_metrics_<mode>_<best_or_last>.json
    results/predictions_<mode>_<best_or_last>.csv
    results/confusion_matrix_<mode>_<best_or_last>.png

Example usage:
    python evaluate.py --mode memeblip2_only
    python evaluate.py --mode context_only
    python evaluate.py --mode context_distillation

Verification run:
    python evaluate.py --mode context_only --verifying
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from tqdm import tqdm

from config import (
    PROJECT_ROOT,
    CHECKPOINT_DIR,
    LOG_DIR,
    RESULTS_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    VALID_TRAIN_MODES,
    NUM_CLASSES,
    SHARED_DIM,
    KNOWLEDGE_DIM,
    NUM_HEADS,
    USE_RESIDUAL_GATE,
    TRAIN_BASE,
    BLIP2_MODEL_NAME,
    CLIP_TEXT_DIM,
    DROPOUT,
    FUSION_TYPE,
    PROJ_LAYERS,
    BETA,
    ensure_project_dirs,
)
from dataset_loader import build_dataloader, move_batch_to_device
from memeblip2 import MemeBLIP2
from knowledge_gated import CrossAttentionKnowledgeMemeBLIP2


# ============================================================
# 1. Model helpers
# ============================================================

def build_model(mode: str, device: str, shared_dim=None, proj_layers=None, beta=None, dropout=None):
    shared_dim = shared_dim if shared_dim is not None else SHARED_DIM
    proj_layers = proj_layers if proj_layers is not None else PROJ_LAYERS
    beta = beta if beta is not None else BETA
    dropout = dropout if dropout is not None else DROPOUT

    if mode == "memeblip2_only":
        model = MemeBLIP2(
            blip2_name=BLIP2_MODEL_NAME,
            shared_dim=shared_dim,
            clip_text_dim=CLIP_TEXT_DIM,
            proj_layers=proj_layers,
            beta=beta,
            dropout=dropout,
            num_classes=NUM_CLASSES,
            fusion_type=FUSION_TYPE,
        )

    elif mode in ["context_only", "context_distillation"]:
        model = CrossAttentionKnowledgeMemeBLIP2(
            blip2_name=BLIP2_MODEL_NAME,
            shared_dim=shared_dim,
            clip_text_dim=CLIP_TEXT_DIM,
            proj_layers=proj_layers,
            beta=beta,
            dropout=dropout,
            knowledge_dim=KNOWLEDGE_DIM,
            num_classes=NUM_CLASSES,
            train_base=TRAIN_BASE,
            num_heads=NUM_HEADS,
            use_residual_gate=USE_RESIDUAL_GATE,
            fusion_type=FUSION_TYPE,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    model = model.to(device)
    return model


def get_checkpoint_path(mode: str, checkpoint_type: str, run_name: str = ""):
    if checkpoint_type not in ["best", "last"]:
        raise ValueError("checkpoint_type must be 'best' or 'last'")

    ckpt_dir = CHECKPOINT_DIR / run_name if run_name else CHECKPOINT_DIR
    return ckpt_dir / f"{mode}_{checkpoint_type}.pt"


def load_checkpoint(model, checkpoint_path: Path, device: str):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train_all_modes.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\n================ Loaded checkpoint ================")
    print("path:", checkpoint_path)
    print("epoch:", checkpoint.get("epoch"))
    print("mode:", checkpoint.get("mode"))
    print("metrics:", checkpoint.get("metrics"))

    return checkpoint


def forward_by_mode(model, batch, mode: str):
    if mode == "memeblip2_only":
        logits = model(
            pixel_values=batch["pixel_values"],
            clip_text_embedding=batch["clip_text_embedding"],
        )
        return {
            "logits": logits,
            "knowledge_attention": None,
            "residual_gate": None,
        }

    if mode in ["context_only", "context_distillation"]:
        outputs = model(
            pixel_values=batch["pixel_values"],
            clip_text_embedding=batch["clip_text_embedding"],
            knowledge_embeddings=batch["knowledge_embeddings"],
            knowledge_mask=batch["knowledge_mask"],
            return_dict=True,
        )
        return outputs

    raise ValueError(f"Unknown mode: {mode}")


# ============================================================
# 2. Metrics and saving
# ============================================================

def compute_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0,
    )

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
    }


def save_metrics(metrics: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_predictions(prediction_rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "text",
        "image_path",
        "source",
        "label",
        "prediction",
        "prob_not_hate",
        "prob_hate",
        "loss",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in prediction_rows:
            writer.writerow(row)


def save_confusion_matrix_plot(cm, output_path: Path, mode: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.array(cm)

    plt.figure()
    plt.imshow(cm)
    plt.title(f"Confusion matrix: {mode}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["not hate", "hate"])
    plt.yticks([0, 1], ["not hate", "hate"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ============================================================
# 3. Evaluation loop
# ============================================================

@torch.no_grad()
def evaluate(model, test_loader, device: str, mode: str, verifying: bool = False):
    model.eval()

    all_labels = []
    all_preds = []
    prediction_rows = []
    total_loss = 0.0
    total_examples = 0

    for step, batch in enumerate(tqdm(test_loader, desc=f"Testing [{mode}]")):
        batch = move_batch_to_device(batch, device)

        labels = batch["labels"].long()
        outputs = forward_by_mode(model, batch, mode)
        logits = outputs["logits"]

        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        losses = F.cross_entropy(logits, labels, reduction="none")

        batch_size = labels.size(0)
        total_loss += losses.sum().item()
        total_examples += batch_size

        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

        ids = batch["id"]
        texts = batch["text"]
        image_paths = batch["image_path"]
        sources = batch["source"]

        probs_cpu = probs.detach().cpu()
        labels_cpu = labels.detach().cpu()
        preds_cpu = preds.detach().cpu()
        losses_cpu = losses.detach().cpu()

        for i in range(batch_size):
            prediction_rows.append({
                "id": ids[i],
                "text": texts[i],
                "image_path": image_paths[i],
                "source": sources[i],
                "label": int(labels_cpu[i].item()),
                "prediction": int(preds_cpu[i].item()),
                "prob_not_hate": float(probs_cpu[i, 0].item()),
                "prob_hate": float(probs_cpu[i, 1].item()),
                "loss": float(losses_cpu[i].item()),
            })

        if verifying and step == 0:
            print("\n================ Evaluation verification batch ================")
            print("logits:", logits.shape, logits.dtype)
            print("probs:", probs.shape, probs.dtype)
            print("clip_text_embedding:", batch["clip_text_embedding"].shape, batch["clip_text_embedding"].dtype)
            print("labels:", labels.shape, labels[:5])
            print("preds:", preds.shape, preds[:5])
            print("losses:", losses.shape, losses[:5])

            if outputs.get("knowledge_attention") is not None:
                print("knowledge_attention:", outputs["knowledge_attention"].shape)

            if outputs.get("residual_gate") is not None:
                print("residual_gate:", outputs["residual_gate"].shape)

            print("\nFirst prediction row:")
            print(prediction_rows[-batch_size])

    metrics = compute_metrics(all_labels, all_preds)
    metrics["test_loss"] = float(total_loss / max(1, total_examples))
    metrics["num_test_examples"] = int(total_examples)

    return metrics, prediction_rows


# ============================================================
# 4. Main
# ============================================================

def run_evaluation(args):
    ensure_project_dirs()

    if args.mode not in VALID_TRAIN_MODES:
        raise ValueError(f"Invalid mode {args.mode}. Choose from {VALID_TRAIN_MODES}")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    print("================ evaluate.py ================")
    print("mode:", args.mode)
    print("checkpoint_type:", args.checkpoint_type)
    print("device:", device)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CHECKPOINT_DIR:", CHECKPOINT_DIR)
    print("RESULTS_DIR:", RESULTS_DIR)
    print("batch_size:", args.batch_size)
    print("num_workers:", args.num_workers)
    print("CLIP_TEXT_DIM:", CLIP_TEXT_DIM)
    print("FUSION_TYPE:", FUSION_TYPE)
    print("DROPOUT:", DROPOUT)
    print("verifying:", args.verifying)

    run_name = getattr(args, "run_name", None) or ""

    # Read architecture params from the run's saved config so we always match
    # the exact model that was trained, regardless of config.py values.
    run_cfg = {}
    if run_name:
        cfg_path = LOG_DIR / run_name / "run_config.json"
        if cfg_path.exists():
            with cfg_path.open() as f:
                run_cfg = json.load(f)
            print("Loaded run_config.json:", cfg_path)

    shared_dim = run_cfg.get("shared_dim", getattr(args, "shared_dim", None))
    proj_layers = run_cfg.get("proj_layers", getattr(args, "proj_layers", None))
    beta = run_cfg.get("beta", getattr(args, "beta", None))
    dropout = run_cfg.get("dropout", None)
    print(f"Architecture: shared_dim={shared_dim or SHARED_DIM}, proj_layers={proj_layers or PROJ_LAYERS}, beta={beta if beta is not None else BETA}, dropout={dropout or DROPOUT}")

    test_loader = build_dataloader(
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        use_teacher=False,
        verifying=args.verifying,
    )

    model = build_model(
        args.mode,
        device,
        shared_dim=shared_dim,
        proj_layers=proj_layers,
        beta=beta,
        dropout=dropout,
    )

    checkpoint_path = get_checkpoint_path(args.mode, args.checkpoint_type, run_name)
    load_checkpoint(model, checkpoint_path, device)

    metrics, prediction_rows = evaluate(
        model=model,
        test_loader=test_loader,
        device=device,
        mode=args.mode,
        verifying=args.verifying,
    )

    run_results_dir = RESULTS_DIR / run_name if run_name else RESULTS_DIR
    suffix = f"{args.mode}_{args.checkpoint_type}"
    metrics_path = run_results_dir / f"test_metrics_{suffix}.json"
    predictions_path = run_results_dir / f"predictions_{suffix}.csv"
    cm_path = run_results_dir / f"confusion_matrix_{suffix}.png"

    save_metrics(metrics, metrics_path)
    save_predictions(prediction_rows, predictions_path)
    save_confusion_matrix_plot(metrics["confusion_matrix"], cm_path, args.mode)

    print("\n================ Test metrics ================")
    print(json.dumps(metrics, indent=2))

    print("\n================ Saved files ================")
    print("Metrics:", metrics_path)
    print("Predictions:", predictions_path)
    print("Confusion matrix:", cm_path)
    print("\n[Done] Evaluation complete.")


# ============================================================
# 5. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=VALID_TRAIN_MODES,
        help="Evaluation mode.",
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
    parser.add_argument("--shared-dim", type=int, default=None)
    parser.add_argument("--proj-layers", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)

    parser.add_argument(
        "--verifying",
        action="store_true",
        help="Print extra checks during first test batch.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Subfolder name under checkpoints/results for this run.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
