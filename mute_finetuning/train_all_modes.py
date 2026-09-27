"""
train_all_modes.py

Train or fine-tune one MUTE experiment mode at a time.

Modes:
    1. memeblip2_only
    2. context_only
    3. context_distillation

This version supports MUTE fine-tuning from previous FHM checkpoints.

The checkpoint paths are controlled from config.py:
    USE_INIT_CHECKPOINTS
    INIT_CHECKPOINTS
    LOAD_INIT_STRICT

For MUTE fine-tuning, model weights are loaded from the FHM checkpoint when
USE_INIT_CHECKPOINTS=True, but the optimizer is restarted by default.

Important:
    For context_distillation:
        - training uses CE + KD loss with teacher_probs
        - validation uses CE-only loss against true labels
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    FORCE_RETRAIN,
    INIT_CHECKPOINTS,
    LEARNING_RATE,
    LOAD_INIT_STRICT,
    LOG_DIR,
    LAMBDA_KD,
    MAX_GRAD_NORM,
    NUM_EPOCHS,
    NUM_WORKERS,
    RANDOM_SEED,
    TEMPERATURE,
    USE_INIT_CHECKPOINTS,
    VALID_TRAIN_MODES,
    VERIFYING,
    WEIGHT_DECAY,
    ensure_project_dirs,
)
from dataset_loader import build_dataloaders, move_batch_to_device
from knowledge_gated import CrossAttentionKnowledgeMemeBLIP2
from memeblip2 import MemeBLIP2


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# 2. Model construction
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
# 3. Checkpoint loading for fine-tuning
# ============================================================

def get_init_checkpoint_for_mode(mode: str):
    return INIT_CHECKPOINTS.get(mode)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint

    raise ValueError("Unsupported checkpoint format.")


def load_initial_checkpoint_if_needed(model, mode: str, device: torch.device):
    """
    Load previous FHM weights before fine-tuning on MUTE.
    """
    if not USE_INIT_CHECKPOINTS:
        print("[Info] USE_INIT_CHECKPOINTS=False, training from scratch.")
        return model

    init_checkpoint_path = get_init_checkpoint_for_mode(mode)

    if init_checkpoint_path is None:
        print(f"[Info] No initial checkpoint configured for mode={mode}.")
        return model

    init_checkpoint_path = Path(init_checkpoint_path)

    if not init_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Initial checkpoint for mode={mode} does not exist:\n"
            f"  {init_checkpoint_path}\n\n"
            "Check FHM_CHECKPOINT_DIR and INIT_CHECKPOINTS in config.py."
        )

    print(f"[Info] Loading initial FHM checkpoint for mode={mode}:")
    print(f"       {init_checkpoint_path}")

    checkpoint = torch.load(init_checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=LOAD_INIT_STRICT,
    )

    print("[Info] Loaded model weights.")
    if not LOAD_INIT_STRICT:
        print(f"[Info] Missing keys: {len(missing_keys)}")
        print(f"[Info] Unexpected keys: {len(unexpected_keys)}")

        if len(missing_keys) > 0:
            print("[Info] First missing keys:", missing_keys[:10])

        if len(unexpected_keys) > 0:
            print("[Info] First unexpected keys:", unexpected_keys[:10])

    print("[Info] Optimizer state is not loaded. Starting fresh optimizer for MUTE.")
    return model


# ============================================================
# 4. Forward pass
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
# 5. Loss functions
# ============================================================

def compute_loss(mode: str, logits, labels, batch, ce_loss_fn):
    """
    memeblip2_only/context_only:
        CE(student, labels)

    context_distillation:
        CE(student, labels) + lambda * KL(student, teacher)
    """
    ce_loss = ce_loss_fn(logits, labels)

    if mode != "context_distillation":
        return ce_loss, {
            "ce_loss": float(ce_loss.detach().cpu().item()),
            "kd_loss": 0.0,
            "total_loss": float(ce_loss.detach().cpu().item()),
        }

    if "teacher_probs" not in batch:
        raise KeyError(
            "context_distillation mode requires teacher_probs in the train batch. "
            "Run generate_teacher_outputs.py or prepare teacher outputs first."
        )

    teacher_probs = batch["teacher_probs"].to(logits.device).float()
    teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    log_student_probs = torch.log_softmax(logits / TEMPERATURE, dim=-1)
    teacher_probs_temp = torch.softmax(
        torch.log(teacher_probs.clamp_min(1e-8)) / TEMPERATURE,
        dim=-1,
    )

    kd_loss = nn.functional.kl_div(
        log_student_probs,
        teacher_probs_temp,
        reduction="batchmean",
    ) * (TEMPERATURE ** 2)

    total_loss = ce_loss + LAMBDA_KD * kd_loss

    return total_loss, {
        "ce_loss": float(ce_loss.detach().cpu().item()),
        "kd_loss": float(kd_loss.detach().cpu().item()),
        "total_loss": float(total_loss.detach().cpu().item()),
    }


# ============================================================
# 6. Metrics
# ============================================================

def compute_epoch_metrics(labels, preds, prob_hateful):
    accuracy = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    try:
        auc = roc_auc_score(labels, prob_hateful)
    except ValueError:
        auc = None

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": None if auc is None else float(auc),
    }


# ============================================================
# 7. Train / validation loops
# ============================================================

def run_one_epoch(
    model,
    dataloader,
    optimizer,
    mode: str,
    device: torch.device,
    train: bool = True,
):
    if train:
        model.train()
    else:
        model.eval()

    ce_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_kd_loss = 0.0
    total_examples = 0

    all_labels = []
    all_preds = []
    all_prob_hateful = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"]

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = forward_batch(model, mode, batch)

            loss, loss_parts = compute_loss(
                mode=mode,
                logits=logits,
                labels=labels,
                batch=batch,
                ce_loss_fn=ce_loss_fn,
            )

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        probs = torch.softmax(logits.detach(), dim=-1)
        preds = torch.argmax(logits.detach(), dim=-1)
        batch_size = labels.size(0)

        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_ce_loss += loss_parts["ce_loss"] * batch_size
        total_kd_loss += loss_parts["kd_loss"] * batch_size
        total_examples += batch_size

        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
        all_prob_hateful.extend(probs[:, 1].detach().cpu().tolist())

    avg_loss = total_loss / max(total_examples, 1)
    avg_ce_loss = total_ce_loss / max(total_examples, 1)
    avg_kd_loss = total_kd_loss / max(total_examples, 1)

    cls_metrics = compute_epoch_metrics(all_labels, all_preds, all_prob_hateful)

    return {
        "loss": float(avg_loss),
        "ce_loss": float(avg_ce_loss),
        "kd_loss": float(avg_kd_loss),
        **cls_metrics,
    }


# ============================================================
# 8. Logging / checkpoints
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch: int,
    mode: str,
    train_metrics: dict,
    val_metrics: dict,
    best_metric_name: str,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "mode": mode,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_metric_name": best_metric_name,
            "config": {
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "lambda_kd": LAMBDA_KD,
                "temperature": TEMPERATURE,
                "use_init_checkpoints": USE_INIT_CHECKPOINTS,
                "load_init_strict": LOAD_INIT_STRICT,
            },
        },
        path,
    )

    print(f"[Info] Saved checkpoint: {path}")


def append_epoch_log(log_path: Path, row: dict):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()

    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_json_summary(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# 9. Main training function
# ============================================================

def train_mode(
    mode: str,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
):
    if mode not in VALID_TRAIN_MODES:
        raise ValueError(f"Invalid mode={mode}. Valid modes: {VALID_TRAIN_MODES}")

    ensure_project_dirs()
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==============================================")
    print(f"Training mode: {mode}")
    print(f"Device: {device}")
    print(f"USE_INIT_CHECKPOINTS: {USE_INIT_CHECKPOINTS}")
    print(f"LOAD_INIT_STRICT: {LOAD_INIT_STRICT}")
    print(f"BATCH_SIZE: {batch_size}")
    print(f"NUM_EPOCHS: {num_epochs}")
    print(f"LEARNING_RATE: {LEARNING_RATE}")
    print(f"LAMBDA_KD: {LAMBDA_KD}")
    print(f"TEMPERATURE: {TEMPERATURE}")
    print("==============================================")

    best_checkpoint_path = CHECKPOINT_DIR / f"{mode}_best.pt"
    last_checkpoint_path = CHECKPOINT_DIR / f"{mode}_last.pt"
    log_path = LOG_DIR / f"train_log_{mode}.csv"
    summary_path = LOG_DIR / f"train_summary_{mode}.json"

    if best_checkpoint_path.exists() and not FORCE_RETRAIN:
        print(f"[Info] Checkpoint already exists and FORCE_RETRAIN=False: {best_checkpoint_path}")
        return

    use_teacher_for_train = mode == "context_distillation"

    train_loader, val_loader, _ = build_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        use_teacher_for_train=use_teacher_for_train,
        verifying=VERIFYING,
    )

    model = build_model(mode, device)
    model = load_initial_checkpoint_if_needed(model=model, mode=mode, device=device)

    optimizer = optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    best_epoch = 0
    best_metric_name = "val_loss"

    history = []

    for epoch in range(1, num_epochs + 1):
        print(f"\n========== Epoch {epoch}/{num_epochs} ==========")

        train_metrics = run_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            mode=mode,
            device=device,
            train=True,
        )

        # For context_distillation, validation should not require teacher_probs.
        # We validate the same context model using supervised CE loss only.
        eval_loss_mode = "context_only" if mode == "context_distillation" else mode

        val_metrics = run_one_epoch(
            model=model,
            dataloader=val_loader,
            optimizer=optimizer,
            mode=eval_loss_mode,
            device=device,
            train=False,
        )

        print(
            f"[Train] loss={train_metrics['loss']:.4f} "
            f"ce={train_metrics['ce_loss']:.4f} "
            f"kd={train_metrics['kd_loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"f1={train_metrics['f1']:.4f}"
        )
        print(
            f"[Val]   loss={val_metrics['loss']:.4f} "
            f"ce={val_metrics['ce_loss']:.4f} "
            f"kd={val_metrics['kd_loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"f1={val_metrics['f1']:.4f}"
        )

        epoch_row = {
            "epoch": epoch,
            "mode": mode,
            "train_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics["ce_loss"],
            "train_kd_loss": train_metrics["kd_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_ce_loss": val_metrics["ce_loss"],
            "val_kd_loss": val_metrics["kd_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
        }

        history.append(epoch_row)
        append_epoch_log(log_path, epoch_row)

        save_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            mode=mode,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_metric_name=best_metric_name,
        )

        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch

            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                mode=mode,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_metric_name=best_metric_name,
            )

    summary = {
        "mode": mode,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_f1_at_best_loss": best_val_f1,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": LEARNING_RATE,
        "lambda_kd": LAMBDA_KD,
        "temperature": TEMPERATURE,
        "use_init_checkpoints": USE_INIT_CHECKPOINTS,
        "init_checkpoint": str(get_init_checkpoint_for_mode(mode)) if USE_INIT_CHECKPOINTS else None,
        "history": history,
    }

    save_json_summary(summary_path, summary)

    print(f"\n[Done] Finished mode={mode}")
    print(f"[Done] Best epoch: {best_epoch}")
    print(f"[Done] Best validation loss: {best_val_loss:.4f}")
    print(f"[Done] Log: {log_path}")
    print(f"[Done] Summary: {summary_path}")


# ============================================================
# 10. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=VALID_TRAIN_MODES,
        help="Training mode to run.",
    )
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--verifying",
        action="store_true",
        help="Accepted for compatibility with run scripts; detailed behavior is controlled by config.py.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_mode(
        mode=args.mode,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
