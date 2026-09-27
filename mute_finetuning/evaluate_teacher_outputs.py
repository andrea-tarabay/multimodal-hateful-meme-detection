"""
evaluate_teacher_outputs.py

Evaluate teacher-model outputs against MUTE train ground-truth labels.

Input:
    teacher_outputs/train_teacher_outputs.jsonl

Outputs:
    results/teacher_metrics/teacher_metrics.json
    results/teacher_metrics/teacher_predictions.csv

Supports both output formats:
    New format:
        true_label, predicted_label, prob_hate

    Backward-compatible format:
        label, teacher_pred_label, teacher_probs
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from config import (
    RESULTS_DIR,
    TRAIN_TEACHER_OUTPUT_PATH,
    ensure_project_dirs,
)


# ============================================================
# 1. Helpers
# ============================================================

def read_jsonl(path: Path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_label(row: dict) -> int:
    if "true_label" in row:
        return int(row["true_label"])
    if "label" in row:
        return int(row["label"])
    raise KeyError("Missing true label field: expected true_label or label")


def get_pred(row: dict):
    if "predicted_label" in row:
        return int(row["predicted_label"])
    if "teacher_pred_label" in row:
        return int(row["teacher_pred_label"])
    return None


def get_prob_hate(row: dict) -> float:
    if "prob_hate" in row:
        return float(row["prob_hate"])
    if "teacher_probs" in row:
        return float(row["teacher_probs"][1])
    return 0.5


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


# ============================================================
# 2. Main
# ============================================================

def main():
    ensure_project_dirs()

    print("================ evaluate_teacher_outputs.py ================")
    print("TRAIN_TEACHER_OUTPUT_PATH:", TRAIN_TEACHER_OUTPUT_PATH)

    if not TRAIN_TEACHER_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Teacher output file not found: {TRAIN_TEACHER_OUTPUT_PATH}"
        )

    rows = read_jsonl(TRAIN_TEACHER_OUTPUT_PATH)

    valid_rows = []
    skipped_parse_failed = 0
    skipped_missing_pred = 0

    for row in rows:
        if row.get("parse_failed", False):
            skipped_parse_failed += 1
            continue

        pred = get_pred(row)
        if pred is None or pred not in [0, 1]:
            skipped_missing_pred += 1
            continue

        valid_rows.append({
            "id": row["id"],
            "image_path": row.get("image_path", ""),
            "text": row.get("text", ""),
            "label": get_label(row),
            "teacher_pred_label": pred,
            "teacher_prob_hateful": get_prob_hate(row),
            "confidence": float(row.get("confidence", 0.5)),
            "target": row.get("target", ""),
            "hateful_cue": row.get("hateful_cue", ""),
            "rationale": row.get("rationale", ""),
            "teacher_external_knowledge_included": row.get("teacher_external_knowledge_included", False),
            "teacher_external_knowledge_count": row.get("teacher_external_knowledge_count", 0),
            "teacher_response": row.get("raw_output", row.get("teacher_response", "")),
        })

    if len(valid_rows) == 0:
        raise ValueError(
            "No valid teacher predictions found. "
            f"skipped_parse_failed={skipped_parse_failed}, "
            f"skipped_missing_pred={skipped_missing_pred}"
        )

    df = pd.DataFrame(valid_rows)

    metrics = compute_metrics(
        labels=df["label"].tolist(),
        preds=df["teacher_pred_label"].tolist(),
        probs=df["teacher_prob_hateful"].tolist(),
    )

    metrics["num_total_rows"] = int(len(rows))
    metrics["num_valid_rows"] = int(len(valid_rows))
    metrics["num_skipped_parse_failed"] = int(skipped_parse_failed)
    metrics["num_skipped_missing_pred"] = int(skipped_missing_pred)

    output_dir = RESULTS_DIR / "teacher_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "teacher_metrics.json"
    predictions_path = output_dir / "teacher_predictions.csv"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    df.to_csv(predictions_path, index=False)

    print("\n================ Teacher metrics ================")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n================ Saved files ================")
    print(metrics_path)
    print(predictions_path)


if __name__ == "__main__":
    main()
