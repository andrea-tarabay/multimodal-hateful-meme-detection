"""
train_all_modes.py

Purpose:
    Train one of three modes:

    1. memeblip2_only
       - Model: MemeBLIP2
       - Inputs: image + precomputed CLIP text embedding
       - Loss: cross entropy with true labels

    2. context_only
       - Model: CrossAttentionKnowledgeMemeBLIP2
       - Inputs: image + precomputed CLIP text embedding + retrieved context
       - Loss: cross entropy with true labels

    3. context_distillation
       - Model: CrossAttentionKnowledgeMemeBLIP2
       - Inputs: image + precomputed CLIP text embedding + retrieved context + Qwen teacher probabilities
       - Loss: cross entropy + lambda_kd * KL distillation loss

Inputs:
    cached_inputs/train/*.pt
    cached_inputs/val/*.pt
    cached_inputs/*/*.pt must include clip_text_embedding
    teacher_outputs/train_teacher_outputs.jsonl   only for context_distillation

Outputs:
    checkpoints/<mode>_best.pt
    checkpoints/<mode>_last.pt
    logs/<mode>_metrics.csv
    plots/<mode>_loss.png

Example usage:
    python train_all_modes.py --mode memeblip2_only
    python train_all_modes.py --mode context_only
    python train_all_modes.py --mode context_distillation

Verification run:
    python train_all_modes.py --mode context_only --verifying --num-epochs 1 --batch-size 1
"""

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import (
    PROJECT_ROOT,
    CHECKPOINT_DIR,
    LOG_DIR,
    PLOT_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    LAMBDA_KD,
    TEMPERATURE,
    RANDOM_SEED,
    VALID_TRAIN_MODES,
    NUM_CLASSES,
    SHARED_DIM,
    KNOWLEDGE_DIM,
    NUM_HEADS,
    USE_RESIDUAL_GATE,
    TRAIN_BASE,
    BLIP2_MODEL_NAME,
    CLIP_TEXT_DIM,
    FUSION_TYPE,
    DROPOUT,
    PROJ_LAYERS,
    BETA,
    ensure_project_dirs,
)
from dataset_loader import build_dataloaders, move_batch_to_device
from memeblip2 import MemeBLIP2
from knowledge_gated import CrossAttentionKnowledgeMemeBLIP2


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic=False + benchmark=True: trades exact reproducibility for
    # speed on the cluster. cuDNN will auto-select the fastest conv algorithm.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# 2. Loss functions
# ============================================================

def compute_hard_loss(logits, labels, label_smoothing: float = 0.1):
    return F.cross_entropy(logits, labels.long(), label_smoothing=label_smoothing)


def compute_distillation_loss(
    student_logits,
    teacher_probs,
    temperature: float,
    labels=None,
    teacher_gate: bool = False,
):
    """
    KL distillation loss using teacher probabilities.

    teacher_probs are expected to be shape [B, 2], already summing to 1.

    If teacher_gate=True, the loss is computed per-sample and zeroed out for
    samples where the teacher prediction disagrees with the ground-truth label.
    This removes the ~21.8% of training samples where the Qwen teacher is
    confidently wrong and would otherwise push the student in the wrong direction.
    """
    teacher_probs = teacher_probs.float()
    teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    if teacher_gate and labels is not None:
        # Per-sample KL, then mask out teacher-wrong samples.
        per_sample = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1) * (temperature * temperature)  # (B,)

        gate = (teacher_probs.argmax(dim=-1) == labels.long()).float()  # 1 where teacher agrees
        n_agree = gate.sum().clamp_min(1.0)
        return (per_sample * gate).sum() / n_agree

    kd_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature * temperature)

    return kd_loss


# ============================================================
# 3. Model builder
# ============================================================

def build_model(mode: str, device: str, args):
    if mode == "memeblip2_only":
        model = MemeBLIP2(
            blip2_name=BLIP2_MODEL_NAME,
            shared_dim=args.shared_dim,
            clip_text_dim=CLIP_TEXT_DIM,
            proj_layers=args.proj_layers,
            beta=args.beta,
            dropout=args.dropout,
            fusion_type=FUSION_TYPE,
            num_classes=NUM_CLASSES,
        )

    elif mode in ["context_only", "context_distillation"]:
        model = CrossAttentionKnowledgeMemeBLIP2(
            blip2_name=BLIP2_MODEL_NAME,
            shared_dim=args.shared_dim,
            clip_text_dim=CLIP_TEXT_DIM,
            proj_layers=args.proj_layers,
            beta=args.beta,
            knowledge_dim=KNOWLEDGE_DIM,
            dropout=args.dropout,
            fusion_type=FUSION_TYPE,
            num_classes=NUM_CLASSES,
            train_base=args.train_base,
            num_heads=NUM_HEADS,
            use_residual_gate=USE_RESIDUAL_GATE,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    model = model.to(device)
    return model


# ============================================================
# 4. Forward helper
# ============================================================

def forward_by_mode(model, batch, mode: str):
    if mode == "memeblip2_only":
        logits = model(
            pixel_values=batch["pixel_values"],
            clip_text_embedding=batch["clip_text_embedding"],
        )
        return {"logits": logits}

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
# 5. Train/eval loops
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    epoch: int,
    mode: str,
    lambda_kd: float,
    temperature: float,
    label_smoothing: float = 0.1,
    verifying: bool = False,
    grad_accum_steps: int = 1,
    teacher_gate: bool = False,
):
    model.train()

    total_loss = 0.0
    total_hard_loss = 0.0
    total_kd_loss = 0.0
    all_labels = []
    all_preds = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [{mode}]")

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(progress_bar):
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"].long()

        outputs = forward_by_mode(model, batch, mode)
        logits = outputs["logits"]

        hard_loss = compute_hard_loss(logits, labels, label_smoothing)

        if mode == "context_distillation":
            teacher_probs = batch["teacher_probs"]
            kd_loss = compute_distillation_loss(
                student_logits=logits,
                teacher_probs=teacher_probs,
                temperature=temperature,
                labels=labels,
                teacher_gate=teacher_gate,
            )
            loss = hard_loss + lambda_kd * kd_loss
        else:
            kd_loss = torch.tensor(0.0, device=device)
            loss = hard_loss

        if torch.isnan(loss):
            raise ValueError("Loss became NaN.")

        (loss / grad_accum_steps).backward()

        is_last_batch = (step + 1) == len(train_loader)
        if (step + 1) % grad_accum_steps == 0 or is_last_batch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_hard_loss += hard_loss.item() * batch_size
        total_kd_loss += kd_loss.item() * batch_size

        preds = torch.argmax(logits, dim=-1)
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

        avg_loss = total_loss / max(1, len(all_labels))
        acc = accuracy_score(all_labels, all_preds)

        progress_bar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "acc": f"{acc:.4f}",
        })

        if verifying and step == 0:
            print("\n================ Training verification batch ================")
            print("logits:", logits.shape, logits.dtype)
            print("clip_text_embedding:", batch["clip_text_embedding"].shape, batch["clip_text_embedding"].dtype)
            print("labels:", labels.shape, labels.dtype, labels[:5])
            print("hard_loss:", hard_loss.item())
            if mode == "context_distillation":
                print("teacher_probs:", batch["teacher_probs"].shape, batch["teacher_probs"][:5])
                print("kd_loss:", kd_loss.item())
            if "knowledge_attention" in outputs and outputs["knowledge_attention"] is not None:
                print("knowledge_attention:", outputs["knowledge_attention"].shape)
            if "residual_gate" in outputs and outputs["residual_gate"] is not None:
                print("residual_gate:", outputs["residual_gate"].shape)

    avg_loss = total_loss / max(1, len(all_labels))
    avg_hard_loss = total_hard_loss / max(1, len(all_labels))
    avg_kd_loss = total_kd_loss / max(1, len(all_labels))

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    return {
        "loss": avg_loss,
        "hard_loss": avg_hard_loss,
        "kd_loss": avg_kd_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


@torch.no_grad()
def evaluate_one_epoch(model, val_loader, device, mode: str):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []

    for batch in tqdm(val_loader, desc=f"Validation [{mode}]"):
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"].long()

        outputs = forward_by_mode(model, batch, mode)
        logits = outputs["logits"]

        loss = compute_hard_loss(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=-1)
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / max(1, len(all_labels))
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    # Mirror train_one_epoch's return shape so callers can handle both uniformly.
    # hard_loss == loss here because there is no KD term during validation.
    return {
        "loss": avg_loss,
        "hard_loss": avg_loss,
        "kd_loss": 0.0,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


# ============================================================
# 6. Saving utilities
# ============================================================

def save_checkpoint(path: Path, model, optimizer, epoch: int, mode: str, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "mode": mode,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def append_metrics_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def plot_losses(history, output_path: Path, mode: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]

    plt.figure()
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss curve: {mode}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ============================================================
# 7. Main training function
# ============================================================

def train_mode(args):
    set_seed(args.seed)
    ensure_project_dirs()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    print("================ train_all_modes.py ================")
    print("mode:", args.mode)
    print("device:", device)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CHECKPOINT_DIR:", CHECKPOINT_DIR)
    print("LOG_DIR:", LOG_DIR)
    print("PLOT_DIR:", PLOT_DIR)
    print("batch_size:", args.batch_size)
    print("num_workers:", args.num_workers)
    print("num_epochs:", args.num_epochs)
    print("learning_rate:", args.learning_rate)
    print("weight_decay:", args.weight_decay)
    print("lambda_kd:", args.lambda_kd)
    print("temperature:", args.temperature)
    print("clip_text_dim:", CLIP_TEXT_DIM)
    print("fusion_type:", FUSION_TYPE)
    print("dropout:", DROPOUT)
    print("verifying:", args.verifying)

    if args.mode not in VALID_TRAIN_MODES:
        raise ValueError(f"Invalid mode {args.mode}. Choose from {VALID_TRAIN_MODES}")

    use_teacher = args.mode == "context_distillation"

    train_loader, val_loader, _ = build_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_teacher_for_train=use_teacher,
        verifying=args.verifying,
    )

    model = build_model(args.mode, device, args)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.eta_min)

    run_log_dir = LOG_DIR / args.run_name if args.run_name else LOG_DIR
    run_ckpt_dir = CHECKPOINT_DIR / args.run_name if args.run_name else CHECKPOINT_DIR
    run_plot_dir = PLOT_DIR / args.run_name if args.run_name else PLOT_DIR

    log_path = run_log_dir / f"{args.mode}_metrics.csv"
    best_path = run_ckpt_dir / f"{args.mode}_best.pt"
    last_path = run_ckpt_dir / f"{args.mode}_last.pt"
    plot_path = run_plot_dir / f"{args.mode}_loss.png"

    # Start a fresh log for this run unless requested otherwise.
    if log_path.exists() and not args.append_log:
        log_path.unlink()

    best_val_macro_f1 = -1.0
    history = []

    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            mode=args.mode,
            lambda_kd=args.lambda_kd,
            temperature=args.temperature,
            label_smoothing=args.label_smoothing,
            verifying=args.verifying,
            grad_accum_steps=args.grad_accum,
            teacher_gate=args.teacher_gate,
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
            mode=args.mode,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"\nEpoch {epoch}/{args.num_epochs} [{args.mode}] lr={current_lr:.2e}")
        print(
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Train acc: {train_metrics['accuracy']:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Train macro-F1: {train_metrics['macro_f1']:.4f}"
        )
        print(
            f"Val loss:   {val_metrics['loss']:.4f} | "
            f"Val acc:   {val_metrics['accuracy']:.4f} | "
            f"Val F1:   {val_metrics['f1']:.4f} | "
            f"Val macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        row = {
            "epoch": epoch,
            "mode": args.mode,
            "train_loss": train_metrics["loss"],
            "train_hard_loss": train_metrics["hard_loss"],
            "train_kd_loss": train_metrics["kd_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_macro_f1": val_metrics["macro_f1"],
        }

        append_metrics_csv(log_path, row)
        history.append(row)

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            mode=args.mode,
            metrics=row,
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                mode=args.mode,
                metrics=row,
            )
            print(f"Saved new best checkpoint: {best_path}")

        plot_losses(history, plot_path, args.mode)
        print(f"Updated loss plot: {plot_path}")

    print("\n================ Training complete ================")
    print("Best val macro-F1:", best_val_macro_f1)
    print("Best checkpoint:", best_path)
    print("Last checkpoint:", last_path)
    print("Log file:", log_path)
    print("Loss plot:", plot_path)


# ============================================================
# 8. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=VALID_TRAIN_MODES,
        help="Training mode.",
    )

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--lambda-kd", type=float, default=LAMBDA_KD)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    parser.add_argument(
        "--verifying",
        action="store_true",
        help="Print extra checks during first batch.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )

    parser.add_argument(
        "--append-log",
        action="store_true",
        help="Append to existing CSV log instead of starting fresh.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Subfolder name under checkpoints/logs/plots for this run.",
    )

    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--shared-dim", type=int, default=SHARED_DIM)
    parser.add_argument("--proj-layers", type=int, default=PROJ_LAYERS)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch size = batch_size * grad_accum.",
    )
    parser.add_argument(
        "--teacher-gate",
        action="store_true",
        default=False,
        help="Gate KD loss per-sample: skip samples where teacher disagrees with ground truth.",
    )
    parser.add_argument(
        "--train-base",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=TRAIN_BASE,
        metavar="BOOL",
        help="Train MemeBLIP2 base projectors/adapters (true/false).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_mode(args)
