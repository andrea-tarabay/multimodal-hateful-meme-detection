"""
run_pipeline.py

Purpose:
    Main project pipeline runner.

It runs the full MemeBLIP2 + context + distillation workflow in order:

    1. prepare_dataset.py
    2. generate_teacher_outputs.py
    3. precompute_inputs.py
    4. train_all_modes.py --mode memeblip2_only
    5. train_all_modes.py --mode context_only
    6. train_all_modes.py --mode context_distillation
    7. evaluate.py --mode memeblip2_only
    8. evaluate.py --mode context_only
    9. evaluate.py --mode context_distillation

The script skips steps whose outputs already exist, unless force flags are enabled.

Recommended config.py additions:

    FORCE_RECOMPUTE_SPLITS = False
    FORCE_RECOMPUTE_TEACHER = False
    FORCE_RECOMPUTE_PREPROCESSING = False
    FORCE_RETRAIN = False
    FORCE_REEVALUATE = False

If these variables are not present in config.py, this script defaults them to False.

Usage:
    python run_pipeline.py

Debug/verification:
    Set VERIFYING=True in config.py before running.

Full run:
    Set VERIFYING=False in config.py before running.
"""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

from config import (
    PROJECT_ROOT,
    TRAIN_ROWS_PATH,
    VAL_ROWS_PATH,
    TEST_ROWS_PATH,
    TRAIN_TEACHER_OUTPUT_PATH,
    CACHE_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    RESULTS_DIR,
    VALID_TRAIN_MODES,
    VERIFYING,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LAMBDA_KD,
    TEMPERATURE,
    ensure_project_dirs,
)

# Optional model/debug values. If they are not defined in config.py, keep
# fallback values so this runner remains usable with older configs.
try:
    from config import CLIP_MODEL_NAME
except ImportError:
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

try:
    from config import CLIP_TEXT_DIM
except ImportError:
    CLIP_TEXT_DIM = 512

try:
    from config import USE_CLIP_TEXT_EMBEDDINGS
except ImportError:
    USE_CLIP_TEXT_EMBEDDINGS = True

try:
    from config import FUSION_TYPE
except ImportError:
    FUSION_TYPE = "concat"

try:
    from config import DROPOUT
except ImportError:
    DROPOUT = 0.3

try:
    from config import SHARED_DIM
except ImportError:
    SHARED_DIM = 1024

try:
    from config import PROJ_LAYERS
except ImportError:
    PROJ_LAYERS = 1

try:
    from config import TRAIN_BASE
except ImportError:
    TRAIN_BASE = True

try:
    from config import BETA
except ImportError:
    BETA = 0.0

# Optional force flags. If they are not defined in config.py, default to False.
try:
    from config import FORCE_RECOMPUTE_SPLITS
except ImportError:
    FORCE_RECOMPUTE_SPLITS = True

try:
    from config import FORCE_RECOMPUTE_TEACHER
except ImportError:
    FORCE_RECOMPUTE_TEACHER = True

try:
    from config import FORCE_RECOMPUTE_PREPROCESSING
except ImportError:
    FORCE_RECOMPUTE_PREPROCESSING = True

try:
    from config import FORCE_RETRAIN
except ImportError:
    FORCE_RETRAIN = True

try:
    from config import FORCE_REEVALUATE
except ImportError:
    FORCE_REEVALUATE = True


# ============================================================
# 1. Helpers
# ============================================================

def save_run_config(run_name: str, args):
    """Write all effective hyperparameters to logs/<run_name>/run_config.json."""
    log_dir = LOG_DIR / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_name": run_name,
        "timestamp": datetime.datetime.now().isoformat(),
        # Training
        "learning_rate": args.learning_rate if args.learning_rate is not None else LEARNING_RATE,
        "num_epochs": args.num_epochs if args.num_epochs is not None else NUM_EPOCHS,
        "batch_size": args.batch_size if args.batch_size is not None else BATCH_SIZE,
        "weight_decay": args.weight_decay if args.weight_decay is not None else WEIGHT_DECAY,
        "eta_min": args.eta_min if args.eta_min is not None else 1e-6,
        # Regularization
        "dropout": args.dropout if args.dropout is not None else DROPOUT,
        "label_smoothing": args.label_smoothing if args.label_smoothing is not None else 0.1,
        # Model architecture
        "shared_dim": args.shared_dim if args.shared_dim is not None else SHARED_DIM,
        "proj_layers": args.proj_layers if args.proj_layers is not None else PROJ_LAYERS,
        "beta": args.beta if args.beta is not None else BETA,
        "fusion_type": FUSION_TYPE,
        "train_base": args.train_base if args.train_base is not None else TRAIN_BASE,
        # Distillation
        "lambda_kd": args.lambda_kd if args.lambda_kd is not None else LAMBDA_KD,
        "temperature": args.temperature if args.temperature is not None else TEMPERATURE,
        # Optimization
        "grad_accum": args.grad_accum if args.grad_accum is not None else 1,
        "teacher_gate": args.teacher_gate,
    }

    path = log_dir / "run_config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Run config saved: {path}")


def run_command(command, step_name: str):
    """
    Run one pipeline command and stop immediately if it fails.
    """
    print("\n" + "=" * 80)
    print(f"Running step: {step_name}")
    print("Command:", " ".join(command))
    print("=" * 80)

    result = subprocess.run(command, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        raise RuntimeError(
            f"Step failed: {step_name}\n"
            f"Command: {' '.join(command)}\n"
            f"Return code: {result.returncode}"
        )


def jsonl_exists_and_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def split_files_ready() -> bool:
    return (
        jsonl_exists_and_nonempty(TRAIN_ROWS_PATH)
        and jsonl_exists_and_nonempty(VAL_ROWS_PATH)
        and jsonl_exists_and_nonempty(TEST_ROWS_PATH)
    )


def teacher_outputs_ready() -> bool:
    if not jsonl_exists_and_nonempty(TRAIN_TEACHER_OUTPUT_PATH):
        return False

    # Check that every cached train sample has a teacher output.
    train_cache_dir = CACHE_DIR / "train"
    if not train_cache_dir.exists():
        return True  # Cache not built yet; cannot validate coverage.

    cached_ids = {p.stem for p in train_cache_dir.glob("*.pt")}
    if not cached_ids:
        return True

    import json

    covered = set()
    with TRAIN_TEACHER_OUTPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                covered.add(str(json.loads(line)["id"]))

    return cached_ids.issubset(covered)


def cached_split_ready(split: str) -> bool:
    split_dir = CACHE_DIR / split
    return split_dir.exists() and any(split_dir.glob("*.pt"))


def cached_inputs_ready() -> bool:
    return (
        cached_split_ready("train")
        and cached_split_ready("val")
        and cached_split_ready("test")
    )


def checkpoint_ready(mode: str, run_name: str = "") -> bool:
    ckpt_dir = CHECKPOINT_DIR / run_name if run_name else CHECKPOINT_DIR
    return (ckpt_dir / f"{mode}_best.pt").exists()


def evaluation_ready(mode: str, run_name: str = "") -> bool:
    res_dir = RESULTS_DIR / run_name if run_name else RESULTS_DIR
    return (
        (res_dir / f"test_metrics_{mode}_best.json").exists()
        and (res_dir / f"predictions_{mode}_best.csv").exists()
        and (res_dir / f"confusion_matrix_{mode}_best.png").exists()
    )


def print_pipeline_summary():
    print("================ run_pipeline.py ================")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("VERIFYING:", VERIFYING)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("NUM_EPOCHS:", NUM_EPOCHS)
    print("LEARNING_RATE:", LEARNING_RATE)
    print("VALID_TRAIN_MODES:", VALID_TRAIN_MODES)
    print("USE_CLIP_TEXT_EMBEDDINGS:", USE_CLIP_TEXT_EMBEDDINGS)
    print("CLIP_MODEL_NAME:", CLIP_MODEL_NAME)
    print("CLIP_TEXT_DIM:", CLIP_TEXT_DIM)
    print("FUSION_TYPE:", FUSION_TYPE)
    print("DROPOUT:", DROPOUT)
    print("FORCE_RECOMPUTE_SPLITS:", FORCE_RECOMPUTE_SPLITS)
    print("FORCE_RECOMPUTE_TEACHER:", FORCE_RECOMPUTE_TEACHER)
    print("FORCE_RECOMPUTE_PREPROCESSING:", FORCE_RECOMPUTE_PREPROCESSING)
    print("FORCE_RETRAIN:", FORCE_RETRAIN)
    print("FORCE_REEVALUATE:", FORCE_REEVALUATE)
    print("\nExpected outputs:")
    print("Splits:", TRAIN_ROWS_PATH, VAL_ROWS_PATH, TEST_ROWS_PATH)
    print("Teacher outputs:", TRAIN_TEACHER_OUTPUT_PATH)
    print("Cached inputs:", CACHE_DIR)
    print("Checkpoints:", CHECKPOINT_DIR)
    print("Results:", RESULTS_DIR)


# ============================================================
# 2. Pipeline steps
# ============================================================

def step_prepare_dataset():
    if split_files_ready() and not FORCE_RECOMPUTE_SPLITS:
        print("\n[Skip] Dataset splits already exist.")
        print("       Set FORCE_RECOMPUTE_SPLITS=True in config.py to rerun.")
        return

    run_command(
        [sys.executable, "prepare_dataset.py"],
        step_name="Prepare dataset splits",
    )


def step_generate_teacher_outputs():
    if teacher_outputs_ready() and not FORCE_RECOMPUTE_TEACHER:
        print("\n[Skip] Teacher outputs already exist.")
        print("       Set FORCE_RECOMPUTE_TEACHER=True in config.py to rerun.")
        return

    run_command(
        [sys.executable, "generate_teacher_outputs.py"],
        step_name="Generate Qwen teacher outputs",
    )


def step_precompute_inputs():
    if cached_inputs_ready() and not FORCE_RECOMPUTE_PREPROCESSING:
        print("\n[Skip] Cached preprocessing inputs already exist.")
        print("       Set FORCE_RECOMPUTE_PREPROCESSING=True in config.py to rerun.")
        return

    run_command(
        [sys.executable, "precompute_inputs.py"],
        step_name="Precompute model inputs and contexts",
    )


def step_train_mode(mode: str, run_name: str, train_overrides: list):
    if checkpoint_ready(mode, run_name) and not FORCE_RETRAIN:
        print(f"\n[Skip] Best checkpoint already exists for mode={mode}.")
        print("       Set FORCE_RETRAIN=True in config.py to rerun training.")
        return

    command = [
        sys.executable,
        "train_all_modes.py",
        "--mode", mode,
        "--run-name", run_name,
    ] + train_overrides

    if VERIFYING:
        command.extend(["--num-epochs", "1", "--batch-size", "1", "--num-workers", "0", "--verifying"])

    run_command(command, step_name=f"Train mode: {mode}")


def step_evaluate_mode(mode: str, run_name: str, eval_overrides: list = None):
    if evaluation_ready(mode, run_name) and not FORCE_REEVALUATE:
        print(f"\n[Skip] Evaluation outputs already exist for mode={mode}.")
        print("       Set FORCE_REEVALUATE=True in config.py to rerun evaluation.")
        return

    command = [
        sys.executable,
        "evaluate.py",
        "--mode", mode,
        "--checkpoint-type", "best",
        "--run-name", run_name,
    ]

    if eval_overrides:
        command.extend(eval_overrides)

    if VERIFYING:
        command.extend(["--batch-size", "1", "--num-workers", "0", "--verifying"])

    run_command(command, step_name=f"Evaluate mode: {mode}")


# ============================================================
# 3. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run the full MemeBLIP2 pipeline.")

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run (used as subfolder under checkpoints/logs/plots/results). "
             "Defaults to a timestamp: MMDD-HHMM.",
    )

    # Hyperparameter overrides forwarded to train_all_modes.py
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--lambda-kd", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--shared-dim", type=int, default=None)
    parser.add_argument("--proj-layers", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--eta-min", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--train-base",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        metavar="BOOL",
        help="Whether to train MemeBLIP2 base projectors/adapters (true/false). "
             "false = only the knowledge cross-attention is trainable.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Gradient accumulation steps. Effective batch size = batch_size * grad_accum.",
    )
    parser.add_argument(
        "--teacher-gate",
        action="store_true",
        default=False,
        help="Gate KD loss per-sample: skip samples where teacher disagrees with ground truth.",
    )

    return parser.parse_args()


# ============================================================
# 4. Main
# ============================================================

def main():
    args = parse_args()

    run_name = args.run_name or datetime.datetime.now().strftime("%m%d-%H%M%S")

    # Build list of hyperparam flags to forward to train_all_modes.py
    train_overrides = []
    if args.learning_rate is not None:
        train_overrides += ["--learning-rate", str(args.learning_rate)]
    if args.num_epochs is not None:
        train_overrides += ["--num-epochs", str(args.num_epochs)]
    if args.lambda_kd is not None:
        train_overrides += ["--lambda-kd", str(args.lambda_kd)]
    if args.weight_decay is not None:
        train_overrides += ["--weight-decay", str(args.weight_decay)]
    if args.batch_size is not None:
        train_overrides += ["--batch-size", str(args.batch_size)]
    if args.dropout is not None:
        train_overrides += ["--dropout", str(args.dropout)]
    if args.shared_dim is not None:
        train_overrides += ["--shared-dim", str(args.shared_dim)]
    if args.proj_layers is not None:
        train_overrides += ["--proj-layers", str(args.proj_layers)]
    if args.beta is not None:
        train_overrides += ["--beta", str(args.beta)]
    if args.label_smoothing is not None:
        train_overrides += ["--label-smoothing", str(args.label_smoothing)]
    if args.eta_min is not None:
        train_overrides += ["--eta-min", str(args.eta_min)]
    if args.temperature is not None:
        train_overrides += ["--temperature", str(args.temperature)]
    if args.train_base is not None:
        train_overrides += ["--train-base", str(args.train_base)]
    if args.grad_accum is not None:
        train_overrides += ["--grad-accum", str(args.grad_accum)]
    if args.teacher_gate:
        train_overrides += ["--teacher-gate"]

    # Architecture flags that must match the checkpoint must be forwarded to evaluate.py
    eval_overrides = []
    if args.shared_dim is not None:
        eval_overrides += ["--shared-dim", str(args.shared_dim)]
    if args.proj_layers is not None:
        eval_overrides += ["--proj-layers", str(args.proj_layers)]
    if args.beta is not None:
        eval_overrides += ["--beta", str(args.beta)]

    ensure_project_dirs()
    save_run_config(run_name, args)
    print_pipeline_summary()
    print("run_name:", run_name)
    if train_overrides:
        print("train_overrides:", train_overrides)

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(f"PROJECT_ROOT does not exist: {PROJECT_ROOT}")

    step_prepare_dataset()
    step_generate_teacher_outputs()
    step_precompute_inputs()

    for mode in VALID_TRAIN_MODES:
        step_train_mode(mode, run_name, train_overrides)

    for mode in VALID_TRAIN_MODES:
        step_evaluate_mode(mode, run_name, eval_overrides)

    print("\n" + "=" * 80)
    print(f"[Done] Full pipeline completed successfully. run_name={run_name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
