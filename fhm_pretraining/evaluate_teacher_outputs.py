"""
evaluate_teacher_outputs.py

Evaluate MemeLens/Qwen teacher outputs against the dataset labels.

Use this to compare teacher prompting with and without external knowledge:

    python3 evaluate_teacher_outputs.py \
      --teacher-output-path /scratch/DL_project/teacher_outputs/train_teacher_outputs_no_ext.jsonl \
      --name teacher_no_external

    python3 evaluate_teacher_outputs.py \
      --teacher-output-path /scratch/DL_project/teacher_outputs/train_teacher_outputs_with_ext.jsonl \
      --name teacher_with_external

By default it reads TRAIN_ROWS_PATH and RESULTS_DIR from config.py.
It saves:
    results/teacher_metrics/<name>_metrics.json
    results/teacher_metrics/<name>_predictions.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

try:
    from config import TRAIN_ROWS_PATH, TRAIN_TEACHER_OUTPUT_PATH, RESULTS_DIR
except Exception:
    TRAIN_ROWS_PATH = Path("processed_splits/train_rows.jsonl")
    TRAIN_TEACHER_OUTPUT_PATH = Path("teacher_outputs/train_teacher_outputs.jsonl")
    RESULTS_DIR = Path("results")


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_labels(rows_path: Path) -> Dict[str, int]:
    rows = read_jsonl(rows_path)
    labels = {}
    for row in rows:
        row_id = str(row["id"])
        labels[row_id] = int(row["label"])
    return labels


def load_teacher_outputs(path: Path) -> Dict[str, dict]:
    rows = read_jsonl(path)
    outputs = {}
    for row in rows:
        row_id = str(row["id"])
        outputs[row_id] = row
    return outputs


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def compute_metrics_for_teacher(
    rows_path: Path,
    teacher_output_path: Path,
    name: str,
    output_dir: Path,
) -> dict:
    labels_by_id = load_labels(rows_path)
    teacher_by_id = load_teacher_outputs(teacher_output_path)

    common_ids = [row_id for row_id in labels_by_id.keys() if row_id in teacher_by_id]
    missing_ids = [row_id for row_id in labels_by_id.keys() if row_id not in teacher_by_id]

    records = []
    y_true_valid = []
    y_pred_valid = []

    strict_correct = 0
    parse_failed_count = 0
    external_knowledge_true_count = 0
    external_knowledge_false_count = 0

    for row_id in common_ids:
        true_label = int(labels_by_id[row_id])
        out = teacher_by_id[row_id]

        pred = out.get("predicted_label", -1)
        try:
            pred = int(pred)
        except Exception:
            pred = -1

        parse_failed = bool(out.get("parse_failed", False)) or pred not in [0, 1]
        if parse_failed:
            parse_failed_count += 1

        if pred == true_label:
            strict_correct += 1

        if pred in [0, 1] and not parse_failed:
            y_true_valid.append(true_label)
            y_pred_valid.append(pred)

        ext_included = out.get("teacher_external_knowledge_included", None)
        if ext_included is True:
            external_knowledge_true_count += 1
        elif ext_included is False:
            external_knowledge_false_count += 1

        records.append({
            "id": row_id,
            "true_label": true_label,
            "predicted_label": pred,
            "correct": int(pred == true_label),
            "parse_failed": parse_failed,
            "prob_not_hate": safe_float(out.get("prob_not_hate")),
            "prob_hate": safe_float(out.get("prob_hate")),
            "confidence": safe_float(out.get("confidence")),
            "target": out.get("target", ""),
            "protected_characteristic": out.get("protected_characteristic", ""),
            "hateful_cue": out.get("hateful_cue", ""),
            "teacher_external_knowledge_included": ext_included,
            "teacher_external_knowledge_count": out.get("teacher_external_knowledge_count", ""),
            "teacher_external_knowledge_status": out.get("teacher_external_knowledge_status", ""),
            "rationale": out.get("rationale", ""),
            "parse_error": out.get("parse_error", ""),
        })

    n_total_rows = len(labels_by_id)
    n_outputs = len(common_ids)
    n_missing = len(missing_ids)
    n_valid = len(y_true_valid)

    strict_accuracy = strict_correct / max(n_outputs, 1)
    coverage = n_outputs / max(n_total_rows, 1)
    valid_rate = n_valid / max(n_outputs, 1)

    if n_valid > 0:
        valid_accuracy = accuracy_score(y_true_valid, y_pred_valid)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_valid,
            y_pred_valid,
            average="binary",
            zero_division=0,
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true_valid,
            y_pred_valid,
            average="macro",
            zero_division=0,
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true_valid,
            y_pred_valid,
            average="weighted",
            zero_division=0,
        )
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1]).tolist()
    else:
        valid_accuracy = 0.0
        precision = recall = f1 = 0.0
        macro_precision = macro_recall = macro_f1 = 0.0
        weighted_precision = weighted_recall = weighted_f1 = 0.0
        cm = [[0, 0], [0, 0]]

    metrics = {
        "name": name,
        "rows_path": str(rows_path),
        "teacher_output_path": str(teacher_output_path),
        "num_labeled_rows": n_total_rows,
        "num_teacher_outputs_matched": n_outputs,
        "num_missing_teacher_outputs": n_missing,
        "coverage": float(coverage),
        "num_parse_failed_or_invalid": parse_failed_count,
        "num_valid_predictions": n_valid,
        "valid_prediction_rate": float(valid_rate),
        "strict_accuracy_counting_parse_failures_as_wrong": float(strict_accuracy),
        "valid_only_accuracy": float(valid_accuracy),
        "valid_only_precision_hate_class": float(precision),
        "valid_only_recall_hate_class": float(recall),
        "valid_only_f1_hate_class": float(f1),
        "valid_only_macro_precision": float(macro_precision),
        "valid_only_macro_recall": float(macro_recall),
        "valid_only_macro_f1": float(macro_f1),
        "valid_only_weighted_precision": float(weighted_precision),
        "valid_only_weighted_recall": float(weighted_recall),
        "valid_only_weighted_f1": float(weighted_f1),
        "valid_only_confusion_matrix_labels_0_1": cm,
        "teacher_external_knowledge_true_count": external_knowledge_true_count,
        "teacher_external_knowledge_false_count": external_knowledge_false_count,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pred_path = output_dir / f"{name}_predictions.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["id"])
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    print("\n================ Teacher evaluation ================")
    print("name:", name)
    print("rows_path:", rows_path)
    print("teacher_output_path:", teacher_output_path)
    print("matched outputs:", n_outputs, "/", n_total_rows)
    print("missing outputs:", n_missing)
    print("parse failed/invalid:", parse_failed_count)
    print("strict accuracy:", round(metrics["strict_accuracy_counting_parse_failures_as_wrong"], 4))
    print("valid-only accuracy:", round(metrics["valid_only_accuracy"], 4))
    print("valid-only precision hate:", round(metrics["valid_only_precision_hate_class"], 4))
    print("valid-only recall hate:", round(metrics["valid_only_recall_hate_class"], 4))
    print("valid-only F1 hate:", round(metrics["valid_only_f1_hate_class"], 4))
    print("valid-only macro F1:", round(metrics["valid_only_macro_f1"], 4))
    print("confusion matrix [0,1]:", cm)
    print("saved metrics:", metrics_path)
    print("saved predictions:", pred_path)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-path", type=Path, default=TRAIN_ROWS_PATH)
    parser.add_argument("--teacher-output-path", type=Path, default=TRAIN_TEACHER_OUTPUT_PATH)
    parser.add_argument("--name", type=str, default="teacher")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "teacher_metrics")
    args = parser.parse_args()

    if not args.rows_path.exists():
        raise FileNotFoundError(f"Rows file not found: {args.rows_path}")
    if not args.teacher_output_path.exists():
        raise FileNotFoundError(f"Teacher output file not found: {args.teacher_output_path}")

    compute_metrics_for_teacher(
        rows_path=args.rows_path,
        teacher_output_path=args.teacher_output_path,
        name=args.name,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
