"""
run_mute_finetune_from_fhm.py

Full runner for the second-stage experiment:

    FHM-trained checkpoints
        -> fine-tune on MUTE Bengali/Bangla memes

It keeps the same three experiment modes as the first-stage FHM pipeline:

    1. memeblip2_only
    2. context_only
    3. context_distillation

Pipeline order:
    1. build_mute_dataset.py
    2. prepare_dataset.py
    3. prepare_teacher_outputs_for_split.py OR generate_teacher_outputs.py
    4. precompute_inputs.py
    5. train_all_modes.py --mode <mode>
    6. evaluate.py --mode <mode> --checkpoint-type best

Teacher-output logic:
    If USE_EXISTING_COMMON_TEACHER_OUTPUTS=True, this runner filters the common
    full-dataset teacher file into teacher_outputs/train_teacher_outputs.jsonl.

    Otherwise, it calls generate_teacher_outputs.py on the current train split.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from config import (
    ALLOW_MISSING_COMMON_TEACHER_OUTPUTS,
    BATCH_SIZE,
    CACHE_DIR,
    CHECKPOINT_DIR,
    COMMON_TEACHER_OUTPUT_PATH,
    DATASET_ROOT,
    EXTERNAL_LEXICON_DIR,
    FORCE_RECOMPUTE_PREPROCESSING,
    FORCE_RECOMPUTE_SPLITS,
    FORCE_RECOMPUTE_TEACHER,
    FORCE_REEVALUATE,
    FORCE_RETRAIN,
    HURTLEX_BN_PATH,
    INIT_CHECKPOINTS,
    LEARNING_RATE,
    MUTE_BALANCE_SCOPE,
    MUTE_ID_PREFIX,
    NUM_EPOCHS,
    PROJECT_ROOT,
    RAW_MUTE_ROOT,
    RESULTS_DIR,
    TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE,
    TRAIN_ROWS_PATH,
    TRAIN_TEACHER_OUTPUT_PATH,
    USE_EXISTING_COMMON_TEACHER_OUTPUTS,
    USE_EXTERNAL_LEXICONS,
    USE_GENERICSKB,
    USE_INIT_CHECKPOINTS,
    VALID_TRAIN_MODES,
    VAL_ROWS_PATH,
    TEST_ROWS_PATH,
    VERIFYING,
    ensure_project_dirs,
    print_config_summary,
)


# ============================================================
# 1. Utility
# ============================================================

def run_command(command, description):
    """
    Run a command inside PROJECT_ROOT and stop immediately if it fails.
    """
    print("\n" + "=" * 80)
    print(description)
    print("=" * 80)
    print("[Command]", " ".join(str(x) for x in command))

    result = subprocess.run([str(x) for x in command], cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {result.returncode}:\n"
            f"{' '.join(str(x) for x in command)}"
        )


def path_exists_and_nonempty(path: Path) -> bool:
    path = Path(path)
    return path.exists() and path.is_file() and path.stat().st_size > 0


def dir_exists_with_files(path: Path, pattern: str = "*") -> bool:
    path = Path(path)
    return path.exists() and path.is_dir() and any(path.glob(pattern))


def read_jsonl_ids(path: Path) -> set[str]:
    ids = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(str(json.loads(line).get("id")))
    return ids


# ============================================================
# 2. Readiness checks
# ============================================================

def standardized_dataset_ready() -> bool:
    return (
        path_exists_and_nonempty(DATASET_ROOT / "dataset.jsonl")
        and dir_exists_with_files(DATASET_ROOT / "images", "*")
    )


def split_files_ready() -> bool:
    return (
        path_exists_and_nonempty(TRAIN_ROWS_PATH)
        and path_exists_and_nonempty(VAL_ROWS_PATH)
        and path_exists_and_nonempty(TEST_ROWS_PATH)
    )


def cached_split_ready(split: str) -> bool:
    return dir_exists_with_files(CACHE_DIR / split, "*.pt")


def cached_inputs_ready() -> bool:
    return cached_split_ready("train") and cached_split_ready("val") and cached_split_ready("test")


def teacher_outputs_ready() -> bool:
    if not path_exists_and_nonempty(TRAIN_TEACHER_OUTPUT_PATH):
        return False

    # If train split exists, require teacher outputs to cover all train ids.
    if TRAIN_ROWS_PATH.exists():
        train_ids = read_jsonl_ids(TRAIN_ROWS_PATH)
        teacher_ids = read_jsonl_ids(TRAIN_TEACHER_OUTPUT_PATH)
        if train_ids and not train_ids.issubset(teacher_ids):
            missing = len(train_ids - teacher_ids)
            print(f"[Info] Teacher outputs missing {missing} train ids.")
            return False

    return True


def checkpoint_ready(mode: str) -> bool:
    return (CHECKPOINT_DIR / f"{mode}_best.pt").exists()


def evaluation_ready(mode: str) -> bool:
    return (
        (RESULTS_DIR / f"test_metrics_{mode}_best.json").exists()
        and (RESULTS_DIR / f"test_predictions_{mode}_best.csv").exists()
    )


def check_required_inputs():
    """
    Check raw MUTE files, Bengali HurtLex, teacher source, and optional FHM checkpoints.
    """
    print("\n========== Checking required inputs ==========")

    required_paths = [
        PROJECT_ROOT,
        RAW_MUTE_ROOT,
        RAW_MUTE_ROOT / "MUTE" / "train_hate.xlsx",
        RAW_MUTE_ROOT / "MUTE" / "valid_hate.xlsx",
        RAW_MUTE_ROOT / "MUTE" / "test_hate.xlsx",
    ]

    if USE_EXTERNAL_LEXICONS or TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE:
        required_paths.append(EXTERNAL_LEXICON_DIR)
        required_paths.append(HURTLEX_BN_PATH)

    if USE_EXISTING_COMMON_TEACHER_OUTPUTS:
        required_paths.append(COMMON_TEACHER_OUTPUT_PATH)

    if USE_INIT_CHECKPOINTS:
        for mode in VALID_TRAIN_MODES:
            if mode not in INIT_CHECKPOINTS:
                raise KeyError(f"INIT_CHECKPOINTS missing mode: {mode}")
            required_paths.append(INIT_CHECKPOINTS[mode])

    missing = []
    for path in required_paths:
        path = Path(path)
        if path.exists():
            print(f"[OK] {path}")
        else:
            missing.append(path)

    if missing:
        print("\n[Error] Missing required paths:")
        for path in missing:
            print(f"  - {path}")
        raise FileNotFoundError(
            "Some required inputs are missing. Fix config.py paths or copy the required files."
        )

    print("[Info] All required inputs found.")


# ============================================================
# 3. Pipeline steps
# ============================================================

def step_build_mute_dataset():
    """
    Convert raw MUTE data to standard dataset.jsonl + images/ format.
    """
    if standardized_dataset_ready() and not FORCE_RECOMPUTE_SPLITS:
        print(f"\n[Skip] Standardized MUTE dataset already exists: {DATASET_ROOT / 'dataset.jsonl'}")
        return

    command = [
        sys.executable,
        "build_mute_dataset.py",
        "--raw-root",
        RAW_MUTE_ROOT,
        "--output-root",
        DATASET_ROOT,
        "--balance-scope",
        MUTE_BALANCE_SCOPE,
        "--id-prefix",
        MUTE_ID_PREFIX,
        "--overwrite",
    ]

    run_command(command, "Step 1: Build standardized MUTE dataset")


def step_prepare_splits():
    """
    Create processed_splits/*.jsonl from DATASET_ROOT/dataset.jsonl.
    """
    if split_files_ready() and not FORCE_RECOMPUTE_SPLITS:
        print("\n[Skip] MUTE processed split files already exist.")
        return

    run_command(
        [sys.executable, "prepare_dataset.py"],
        "Step 2: Prepare MUTE train/val/test rows",
    )


def step_prepare_teacher_outputs_from_common():
    """
    Filter common full-dataset teacher outputs to the current train split.
    """
    if teacher_outputs_ready() and not FORCE_RECOMPUTE_TEACHER:
        print(f"\n[Skip] Train teacher outputs already exist and cover train split: {TRAIN_TEACHER_OUTPUT_PATH}")
        return

    command = [
        sys.executable,
        "prepare_teacher_outputs_for_split.py",
        "--common-teacher-output-path",
        COMMON_TEACHER_OUTPUT_PATH,
        "--train-rows-path",
        TRAIN_ROWS_PATH,
        "--output-path",
        TRAIN_TEACHER_OUTPUT_PATH,
    ]

    if FORCE_RECOMPUTE_TEACHER:
        command.append("--force")

    if ALLOW_MISSING_COMMON_TEACHER_OUTPUTS:
        command.append("--allow-missing")

    run_command(
        command,
        "Step 3: Prepare train teacher outputs from common teacher file",
    )


def step_generate_teacher_outputs():
    """
    Generate teacher outputs for the MUTE train split from scratch.
    """
    if teacher_outputs_ready() and not FORCE_RECOMPUTE_TEACHER:
        print(f"\n[Skip] Teacher outputs already exist and cover train split: {TRAIN_TEACHER_OUTPUT_PATH}")
        return

    run_command(
        [sys.executable, "generate_teacher_outputs.py"],
        "Step 3: Generate teacher outputs for MUTE train split",
    )


def step_prepare_or_generate_teacher_outputs():
    """
    Select between existing common teacher outputs and fresh teacher inference.
    """
    if USE_EXISTING_COMMON_TEACHER_OUTPUTS:
        step_prepare_teacher_outputs_from_common()
    else:
        step_generate_teacher_outputs()


def step_precompute_inputs():
    """
    Precompute BLIP2 image pixels, CLIP text embeddings, and context tensors.
    """
    if cached_inputs_ready() and not FORCE_RECOMPUTE_PREPROCESSING:
        print(f"\n[Skip] Cached preprocessing inputs already exist: {CACHE_DIR}")
        return

    run_command(
        [sys.executable, "precompute_inputs.py"],
        "Step 4: Precompute MUTE model inputs",
    )


def step_train_modes():
    """
    Fine-tune all three modes on MUTE.
    """
    for mode in VALID_TRAIN_MODES:
        best_ckpt = CHECKPOINT_DIR / f"{mode}_best.pt"

        if checkpoint_ready(mode) and not FORCE_RETRAIN:
            print(f"\n[Skip] Checkpoint already exists for mode={mode}: {best_ckpt}")
            continue

        command = [sys.executable, "train_all_modes.py", "--mode", mode]

        if VERIFYING:
            command.extend([
                "--num-epochs",
                "1",
                "--batch-size",
                "1",
                "--num-workers",
                "0",
                "--verifying",
            ])

        run_command(command, f"Step 5: Fine-tune mode={mode} on MUTE")


def step_evaluate_modes():
    """
    Evaluate all MUTE fine-tuned models.
    """
    for mode in VALID_TRAIN_MODES:
        if evaluation_ready(mode) and not FORCE_REEVALUATE:
            print(f"\n[Skip] Evaluation already exists for mode={mode}.")
            continue

        command = [
            sys.executable,
            "evaluate.py",
            "--mode",
            mode,
            "--checkpoint-type",
            "best",
        ]

        if VERIFYING:
            command.extend(["--batch-size", "1", "--num-workers", "0", "--verifying"])

        run_command(command, f"Step 6: Evaluate fine-tuned mode={mode} on MUTE test split")


# ============================================================
# 4. Main
# ============================================================

def main():
    print("\n============================================================")
    print("MUTE fine-tuning pipeline from FHM checkpoints")
    print("============================================================")

    ensure_project_dirs()
    print_config_summary()

    print("\n========== Run summary ==========")
    print("VERIFYING:", VERIFYING)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("NUM_EPOCHS:", NUM_EPOCHS)
    print("LEARNING_RATE:", LEARNING_RATE)
    print("VALID_TRAIN_MODES:", VALID_TRAIN_MODES)

    print("\n========== Teacher output settings ==========")
    print("USE_EXISTING_COMMON_TEACHER_OUTPUTS:", USE_EXISTING_COMMON_TEACHER_OUTPUTS)
    print("COMMON_TEACHER_OUTPUT_PATH:", COMMON_TEACHER_OUTPUT_PATH)
    print("TRAIN_TEACHER_OUTPUT_PATH:", TRAIN_TEACHER_OUTPUT_PATH)
    print("ALLOW_MISSING_COMMON_TEACHER_OUTPUTS:", ALLOW_MISSING_COMMON_TEACHER_OUTPUTS)

    print("\n========== Context settings ==========")
    print("USE_GENERICSKB:", USE_GENERICSKB)
    print("USE_EXTERNAL_LEXICONS:", USE_EXTERNAL_LEXICONS)
    print("TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE:", TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE)
    print("EXTERNAL_LEXICON_DIR:", EXTERNAL_LEXICON_DIR)
    print("HURTLEX_BN_PATH:", HURTLEX_BN_PATH)

    print("\n========== Initial FHM checkpoints ==========")
    print("USE_INIT_CHECKPOINTS:", USE_INIT_CHECKPOINTS)
    for mode in VALID_TRAIN_MODES:
        print(f"{mode}: {INIT_CHECKPOINTS.get(mode)}")

    check_required_inputs()

    step_build_mute_dataset()
    step_prepare_splits()
    step_prepare_or_generate_teacher_outputs()
    step_precompute_inputs()
    step_train_modes()
    step_evaluate_modes()

    print("\n============================================================")
    print("MUTE fine-tuning pipeline finished successfully.")
    print("============================================================")


if __name__ == "__main__":
    main()
