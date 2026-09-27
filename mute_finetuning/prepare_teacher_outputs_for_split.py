"""
prepare_teacher_outputs_for_split.py

Prepare train-split teacher outputs from a common teacher-output JSONL file.

Use case:
    Teacher outputs already generated for the full MUTE dataset and saved
    them in one shared/common JSONL file instead of separate train/val/test files.

This script reads:
    COMMON_TEACHER_OUTPUT_PATH
    processed_splits/train_rows.jsonl

and writes:
    teacher_outputs/train_teacher_outputs.jsonl

Only train teacher outputs are required for distillation. Validation/test should
use ground-truth labels only.

Matching priority:
    1. id
    2. image_path
    3. original_image_name

The output is normalized to the format expected by dataset_loader.py and
train_all_modes.py:
    id
    image_path
    text
    true_label
    label
    source
    prob_not_hate
    prob_hate
    teacher_probs
    predicted_label
    teacher_pred_label
    confidence
    target
    hateful_cue
    rationale
    raw_output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from config import (
    COMMON_TEACHER_OUTPUT_PATH,
    FORCE_RECOMPUTE_TEACHER,
    TRAIN_ROWS_PATH,
    TRAIN_TEACHER_OUTPUT_PATH,
    ensure_project_dirs,
)


# ============================================================
# 1. JSONL helpers
# ============================================================

def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_key(value) -> str:
    return str(value).strip().lower().replace("\\", "/")


def basename_key(value) -> str:
    return Path(str(value).replace("\\", "/")).name.strip().lower()


# ============================================================
# 2. Teacher-output normalization
# ============================================================

def get_probabilities(row: dict) -> Optional[Tuple[float, float]]:
    """
    Return (prob_not_hate, prob_hate) from supported teacher-output formats.
    """
    if "prob_not_hate" in row and "prob_hate" in row:
        prob_not_hate = float(row["prob_not_hate"])
        prob_hate = float(row["prob_hate"])
    elif "teacher_probs" in row:
        teacher_probs = row["teacher_probs"]
        prob_not_hate = float(teacher_probs[0])
        prob_hate = float(teacher_probs[1])
    else:
        return None

    total = prob_not_hate + prob_hate
    if total <= 0:
        return None

    prob_not_hate = prob_not_hate / total
    prob_hate = prob_hate / total
    return float(prob_not_hate), float(prob_hate)


def get_predicted_label(row: dict, prob_not_hate: float, prob_hate: float) -> int:
    if "predicted_label" in row:
        try:
            pred = int(row["predicted_label"])
            if pred in [0, 1]:
                return pred
        except Exception:
            pass

    if "teacher_pred_label" in row:
        try:
            pred = int(row["teacher_pred_label"])
            if pred in [0, 1]:
                return pred
        except Exception:
            pass

    return int(prob_hate >= prob_not_hate)


def normalize_teacher_row(teacher_row: dict, train_row: dict) -> Optional[dict]:
    """
    Normalize one common teacher row so it is compatible with training.
    """
    probs = get_probabilities(teacher_row)
    if probs is None:
        return None

    prob_not_hate, prob_hate = probs
    predicted_label = get_predicted_label(teacher_row, prob_not_hate, prob_hate)

    output = {
        "id": train_row["id"],
        "image_path": train_row["image_path"],
        "text": train_row.get("text", teacher_row.get("text", "")),
        "true_label": int(train_row["label"]),
        "label": int(train_row["label"]),
        "source": train_row.get("source", teacher_row.get("source", "mute")),
        "prob_not_hate": round(float(prob_not_hate), 6),
        "prob_hate": round(float(prob_hate), 6),
        "teacher_probs": [round(float(prob_not_hate), 6), round(float(prob_hate), 6)],
        "predicted_label": predicted_label,
        "teacher_pred_label": predicted_label,
        "confidence": float(teacher_row.get("confidence", max(prob_not_hate, prob_hate))),
        "target": teacher_row.get("target", "unclear"),
        "protected_characteristic": teacher_row.get("protected_characteristic", "not_required_for_mute"),
        "hateful_cue": teacher_row.get("hateful_cue", "none"),
        "rationale": teacher_row.get("rationale", teacher_row.get("explanation", "")),
        "parse_failed": bool(teacher_row.get("parse_failed", False)),
        "parse_error": teacher_row.get("parse_error", None),
        "raw_output": teacher_row.get("raw_output", teacher_row.get("teacher_response", "")),
        "teacher_response": teacher_row.get("teacher_response", teacher_row.get("raw_output", "")),
        "teacher_external_knowledge_included": bool(teacher_row.get("teacher_external_knowledge_included", False)),
        "teacher_external_knowledge_status": teacher_row.get("teacher_external_knowledge_status", "from_common_file"),
        "teacher_external_knowledge_count": int(teacher_row.get("teacher_external_knowledge_count", 0) or 0),
        "teacher_external_knowledge_text": teacher_row.get("teacher_external_knowledge_text", ""),
        "matched_from_common_teacher_file": True,
    }

    for optional_key in ["original_split", "original_image_name", "original_row_index"]:
        if optional_key in train_row:
            output[optional_key] = train_row[optional_key]

    if "id" in teacher_row:
        output["common_teacher_id"] = teacher_row["id"]
    if "image_path" in teacher_row:
        output["common_teacher_image_path"] = teacher_row["image_path"]
    if "original_image_name" in teacher_row:
        output["common_teacher_original_image_name"] = teacher_row["original_image_name"]

    return output


# ============================================================
# 3. Matching
# ============================================================

def build_teacher_indexes(common_rows: List[dict]) -> Dict[str, Dict[str, dict]]:
    """
    Build lookup tables over the common teacher-output file.
    """
    indexes = {
        "id": {},
        "image_path": {},
        "image_basename": {},
        "original_image_name": {},
    }

    duplicate_counts = {key: 0 for key in indexes}

    def add(index_name: str, key: str, row: dict):
        if not key:
            return
        if key in indexes[index_name]:
            duplicate_counts[index_name] += 1
            return
        indexes[index_name][key] = row

    for row in common_rows:
        if "id" in row:
            add("id", normalize_key(row["id"]), row)
        if "image_path" in row:
            add("image_path", normalize_key(row["image_path"]), row)
            add("image_basename", basename_key(row["image_path"]), row)
        if "original_image_name" in row:
            add("original_image_name", basename_key(row["original_image_name"]), row)
        if "original_image_name" not in row and "image_name" in row:
            add("original_image_name", basename_key(row["image_name"]), row)

    print("\nTeacher index sizes:")
    for name, index in indexes.items():
        print(f"  {name}: {len(index)} unique keys, {duplicate_counts[name]} duplicate keys ignored")

    return indexes


def find_teacher_row(train_row: dict, indexes: Dict[str, Dict[str, dict]]) -> Tuple[Optional[dict], str]:
    """
    Match a train row against the common teacher-output indexes.
    """
    row_id = normalize_key(train_row.get("id", ""))
    if row_id and row_id in indexes["id"]:
        return indexes["id"][row_id], "id"

    image_path = normalize_key(train_row.get("image_path", ""))
    if image_path and image_path in indexes["image_path"]:
        return indexes["image_path"][image_path], "image_path"

    image_base = basename_key(train_row.get("image_path", ""))
    if image_base and image_base in indexes["image_basename"]:
        return indexes["image_basename"][image_base], "image_basename"

    original_name = basename_key(train_row.get("original_image_name", ""))
    if original_name and original_name in indexes["original_image_name"]:
        return indexes["original_image_name"][original_name], "original_image_name"

    return None, "no_match"


# ============================================================
# 4. Main conversion
# ============================================================

def prepare_teacher_outputs(
    common_teacher_output_path: Path,
    train_rows_path: Path,
    output_path: Path,
    force: bool = False,
    allow_missing: bool = False,
):
    common_teacher_output_path = Path(common_teacher_output_path)
    train_rows_path = Path(train_rows_path)
    output_path = Path(output_path)

    if output_path.exists() and not force:
        print(f"[Skip] Output already exists and force=False: {output_path}")
        return

    if not common_teacher_output_path.exists():
        raise FileNotFoundError(f"Common teacher-output file not found: {common_teacher_output_path}")

    if not train_rows_path.exists():
        raise FileNotFoundError(f"Train rows file not found: {train_rows_path}")

    common_rows = read_jsonl(common_teacher_output_path)
    train_rows = read_jsonl(train_rows_path)

    print("================ prepare_teacher_outputs_for_split.py ================")
    print("COMMON_TEACHER_OUTPUT_PATH:", common_teacher_output_path)
    print("TRAIN_ROWS_PATH:", train_rows_path)
    print("OUTPUT_PATH:", output_path)
    print("Common teacher rows:", len(common_rows))
    print("Train rows:", len(train_rows))
    print("force:", force)
    print("allow_missing:", allow_missing)

    indexes = build_teacher_indexes(common_rows)

    matched_rows = []
    missing_rows = []
    unusable_rows = []
    match_counts = {}

    for train_row in train_rows:
        teacher_row, match_method = find_teacher_row(train_row, indexes)
        match_counts[match_method] = match_counts.get(match_method, 0) + 1

        if teacher_row is None:
            missing_rows.append(train_row)
            continue

        normalized = normalize_teacher_row(teacher_row, train_row)
        if normalized is None:
            unusable_rows.append(train_row)
            continue

        normalized["teacher_match_method"] = match_method
        matched_rows.append(normalized)

    print("\nMatch summary:")
    for method, count in sorted(match_counts.items()):
        print(f"  {method}: {count}")

    print("\nOutput summary:")
    print("Matched usable rows:", len(matched_rows))
    print("Missing rows:", len(missing_rows))
    print("Unusable teacher rows:", len(unusable_rows))

    if missing_rows:
        print("\nFirst missing examples:")
        for row in missing_rows[:10]:
            print("  id:", row.get("id"), "image_path:", row.get("image_path"), "original_image_name:", row.get("original_image_name"))

    if unusable_rows:
        print("\nFirst unusable examples:")
        for row in unusable_rows[:10]:
            print("  id:", row.get("id"), "image_path:", row.get("image_path"))

    if (missing_rows or unusable_rows) and not allow_missing:
        raise ValueError(
            "Could not prepare complete train teacher outputs. "
            f"missing={len(missing_rows)}, unusable={len(unusable_rows)}. "
            "Use --allow-missing only for debugging, not final training."
        )

    write_jsonl(output_path, matched_rows)
    print("\n[Done] Saved train teacher outputs:", output_path)


# ============================================================
# 5. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--common-teacher-output-path",
        type=Path,
        default=COMMON_TEACHER_OUTPUT_PATH,
    )
    parser.add_argument(
        "--train-rows-path",
        type=Path,
        default=TRAIN_ROWS_PATH,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=TRAIN_TEACHER_OUTPUT_PATH,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=FORCE_RECOMPUTE_TEACHER,
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing/unusable teacher rows. Use only for debugging.",
    )

    return parser.parse_args()


def main():
    ensure_project_dirs()
    args = parse_args()
    prepare_teacher_outputs(
        common_teacher_output_path=args.common_teacher_output_path,
        train_rows_path=args.train_rows_path,
        output_path=args.output_path,
        force=args.force,
        allow_missing=args.allow_missing,
    )


if __name__ == "__main__":
    main()
