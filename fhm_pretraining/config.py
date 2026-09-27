"""
config.py

Central configuration file for the MemeBLIP2 + context + distillation project.

Use this file as the single source of truth for:
    - cluster paths
    - dataset/cache/checkpoint locations
    - verifying/debug settings
    - training hyperparameters
    - distillation hyperparameters

"""

from pathlib import Path


# ============================================================
# 1. Cluster paths
# ============================================================

SCRATCH_ROOT = Path("/scratch")
PROJECT_ROOT = SCRATCH_ROOT / "DL_project_alexis"
DATASET_ROOT = SCRATCH_ROOT / "meme_dataset_separated" / "fhm"
HF_CACHE_DIR = SCRATCH_ROOT / "hf_cache" / "hub"


# ============================================================
# 2. Project folders
# ============================================================

SPLIT_DIR = PROJECT_ROOT / "processed_splits"
CACHE_DIR = PROJECT_ROOT / "cached_inputs"
TEACHER_OUTPUT_DIR = PROJECT_ROOT / "teacher_outputs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
PLOT_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# 3. Split files
# ============================================================

TRAIN_ROWS_PATH = SPLIT_DIR / "train_rows.jsonl"
VAL_ROWS_PATH = SPLIT_DIR / "val_rows.jsonl"
TEST_ROWS_PATH = SPLIT_DIR / "test_rows.jsonl"


# ============================================================
# 4. Teacher output files
# ============================================================

TRAIN_TEACHER_OUTPUT_PATH = TEACHER_OUTPUT_DIR / "train_teacher_outputs.jsonl"


# ============================================================
# 5. Dataset settings
# ============================================================

REQUIRED_COLUMNS = ["id", "image_path", "text", "label", "source"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

LABEL_NOT_HATE = 0
LABEL_HATE = 1


# ============================================================
# 6. Verification/debug settings
# ============================================================

# Turn this on for tiny one-row or few-row tests.
VERIFYING = False
VERIFYING_NUM_ROWS = 1000
BALANCE_CLASSES = True

# ============================================================
# 7. Model settings
# ============================================================

BLIP2_MODEL_NAME = "Salesforce/blip2-opt-2.7b"
TEACHER_MODEL_NAME = "QCRI/MemeLens-VLM"  # before "Qwen/Qwen3-VL-4B-Instruct"

NUM_CLASSES = 2
SHARED_DIM = 1024
KNOWLEDGE_DIM = 384
KNOWLEDGE_TOP_K = 5
NUM_HEADS = 8
USE_RESIDUAL_GATE = True
TRAIN_BASE = True

# CLIP text embeddings replace the previous randomly initialized
# MemeBLIP2 text embedding branch.
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_TEXT_DIM = 512
USE_CLIP_TEXT_EMBEDDINGS = True

# Fusion used after projecting image and CLIP-text features to SHARED_DIM.
# "concat" is safer than element-wise multiplication when the two encoders
# come from different pretrained models.
FUSION_TYPE = "concat"

# Dropout used in MemeBLIP2 and knowledge-gated trainable modules.
DROPOUT = 0.3

# Linear projector depth (1 = single layer, no residual shortcut).
PROJ_LAYERS = 1

# Adapter mix ratio: 0.0 disables the feature adapters entirely.
BETA = 0.2


# ============================================================
# 8. Training hyperparameters
# ============================================================

BATCH_SIZE = 4
NUM_WORKERS = 2
NUM_EPOCHS = 15

# Cosine scheduler decays this to 1e-6 over NUM_EPOCHS.
LEARNING_RATE = 3e-5

WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
RANDOM_SEED = 42


# ============================================================
# 9. Distillation hyperparameters
# ============================================================

LAMBDA_KD = 0.15
TEMPERATURE = 2.0


# ============================================================
# 10. Qwen generation settings
# ============================================================

MAX_NEW_TOKENS = 256
USE_FLASH_ATTENTION_2 = False

TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE = True
TEACHER_EXTERNAL_KNOWLEDGE_MAX_ITEMS = 5
TEACHER_EXTERNAL_KNOWLEDGE_MIN_SCORE = 0.55


# ============================================================
# 11. Valid modes
# ============================================================

VALID_TRAIN_MODES = [
    "memeblip2_only",
    "context_only",
    "context_distillation",
]


# ============================================================
# 12. Utility
# ============================================================

def ensure_project_dirs():
    """
    Create all standard output folders if they do not exist.
    """
    for path in [
        SPLIT_DIR,
        CACHE_DIR,
        TEACHER_OUTPUT_DIR,
        CHECKPOINT_DIR,
        LOG_DIR,
        PLOT_DIR,
        RESULTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def print_config_summary():
    """
    Print the most important config values.
    Useful for debugging on the cluster.
    """
    print("================ config.py summary ================")
    print("SCRATCH_ROOT:", SCRATCH_ROOT)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATASET_ROOT:", DATASET_ROOT)
    print("HF_CACHE_DIR:", HF_CACHE_DIR)
    print("SPLIT_DIR:", SPLIT_DIR)
    print("CACHE_DIR:", CACHE_DIR)
    print("TEACHER_OUTPUT_DIR:", TEACHER_OUTPUT_DIR)
    print("CHECKPOINT_DIR:", CHECKPOINT_DIR)
    print("LOG_DIR:", LOG_DIR)
    print("PLOT_DIR:", PLOT_DIR)
    print("RESULTS_DIR:", RESULTS_DIR)
    print("VERIFYING:", VERIFYING)
    print("VERIFYING_NUM_ROWS:", VERIFYING_NUM_ROWS)
    print("FORCE_RECOMPUTE:", FORCE_RECOMPUTE)
    print("BLIP2_MODEL_NAME:", BLIP2_MODEL_NAME)
    print("CLIP_MODEL_NAME:", CLIP_MODEL_NAME)
    print("CLIP_TEXT_DIM:", CLIP_TEXT_DIM)
    print("USE_CLIP_TEXT_EMBEDDINGS:", USE_CLIP_TEXT_EMBEDDINGS)
    print("FUSION_TYPE:", FUSION_TYPE)
    print("DROPOUT:", DROPOUT)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("NUM_EPOCHS:", NUM_EPOCHS)
    print("LEARNING_RATE:", LEARNING_RATE)
    print("WEIGHT_DECAY:", WEIGHT_DECAY)
    print("LAMBDA_KD:", LAMBDA_KD)
    print("TEMPERATURE:", TEMPERATURE)


if __name__ == "__main__":
    ensure_project_dirs()
    print_config_summary()


# ============================================================
# 13. Pipeline force flags
# ============================================================

FORCE_RECOMPUTE = False

FORCE_RECOMPUTE_SPLITS = False

FORCE_RECOMPUTE_TEACHER = False

FORCE_RECOMPUTE_PREPROCESSING = False

FORCE_RETRAIN = True

FORCE_REEVALUATE = True
