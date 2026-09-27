"""
config.py

Central configuration file for the second-stage MUTE fine-tuning pipeline.

This stage fine-tunes the same three MemeBLIP2 experiments on the Bengali/Bangla
MUTE dataset, optionally starting from FHM-trained checkpoints:

    1. memeblip2_only
    2. context_only
    3. context_distillation

"""

from pathlib import Path


# ============================================================
# 1. Cluster paths
# ============================================================

SCRATCH_ROOT = Path("/scratch")

# Folder containing this MUTE-stage code on the cluster.
PROJECT_ROOT = SCRATCH_ROOT / "finetune-joelle"

# Raw MUTE folder before conversion.
RAW_MUTE_ROOT = SCRATCH_ROOT / "mute_raw"

# Balanced standardized MUTE dataset matching the existing teacher outputs.
DATASET_ROOT = SCRATCH_ROOT / "meme_dataset_separated" / "mute_balanced_3172"

# Hugging Face cache on the cluster.
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

# Bengali/Bangla-only external lexicon folder.
EXTERNAL_LEXICON_DIR = PROJECT_ROOT / "external_lexicons_bn_only"
HURTLEX_BN_PATH = EXTERNAL_LEXICON_DIR / "hurtlex_BN.tsv"


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

USE_EXISTING_COMMON_TEACHER_OUTPUTS = True
COMMON_TEACHER_OUTPUT_PATH = (
    SCRATCH_ROOT
    / "DL_project_mute_teacher50"
    / "teacher_outputs"
    / "prompt_runs"
    / "FINAL_mute_noext_teacher_outputs_ALL_mute_balanced3172.jsonl"
)

ALLOW_MISSING_COMMON_TEACHER_OUTPUTS = False


# ============================================================
# 5. Dataset settings
# ============================================================

REQUIRED_COLUMNS = ["id", "image_path", "text", "label", "source"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

LABEL_NOT_HATE = 0
LABEL_HATE = 1

MUTE_LABEL_MAP = {
    "not-hate": LABEL_NOT_HATE,
    "hate": LABEL_HATE,
}

BALANCE_CLASSES = False
MUTE_BALANCE_SCOPE = "combined"
MUTE_ID_PREFIX = "mute_balanced"


# ============================================================
# 6. Verification/debug settings
# ============================================================

VERIFYING = False
VERIFYING_NUM_ROWS = 1000

FORCE_RECOMPUTE = False


# ============================================================
# 7. Model settings
# ============================================================

BLIP2_MODEL_NAME = "Salesforce/blip2-opt-2.7b"
TEACHER_MODEL_NAME = "QCRI/MemeLens-VLM"

NUM_CLASSES = 2
SHARED_DIM = 512
KNOWLEDGE_DIM = 384
KNOWLEDGE_TOP_K = 5
NUM_HEADS = 8
USE_RESIDUAL_GATE = True
TRAIN_BASE = True

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_TEXT_DIM = 512
USE_CLIP_TEXT_EMBEDDINGS = True

FUSION_TYPE = "concat"
DROPOUT = 0.5
PROJ_LAYERS = 1
BETA = 0.0

# Present in the FHM run_config.json.
# These only affect training/model behavior if the relevant code imports/uses them.
TEACHER_GATE = True


# ============================================================
# 8. Training hyperparameters
# ============================================================

BATCH_SIZE = 4
NUM_WORKERS = 2
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
RANDOM_SEED = 42

# Present in the FHM run_config.json.
# These only affect training if train_all_modes.py imports/uses them.
ETA_MIN = 1e-6
LABEL_SMOOTHING = 0.15
GRAD_ACCUM = 8


# ============================================================
# 9. Distillation hyperparameters
# ============================================================

LAMBDA_KD = 0.1
TEMPERATURE = 2.0


# ============================================================
# 10. Qwen/MemeLens teacher generation settings
# ============================================================

MAX_NEW_TOKENS = 256
USE_FLASH_ATTENTION_2 = False

TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE = True
TEACHER_EXTERNAL_KNOWLEDGE_MAX_ITEMS = 5
TEACHER_EXTERNAL_KNOWLEDGE_MIN_SCORE = 0.55


# ============================================================
# 11. External knowledge retrieval settings
# ============================================================

USE_EXTERNAL_LEXICONS = True
USE_GENERICSKB = False
USE_WIKIDATA = False
USE_WIKIPEDIA = False

EXTERNAL_LEXICON_RESULTS_PER_TERM = 20
EXTERNAL_LEXICON_MIN_SCORE = 0.55

ALPHA_BM25 = 0.45
BETA_EMBEDDING = 0.50
GAMMA_SOURCE = 0.05

NORMALIZE_KNOWLEDGE_EMBEDDINGS = True


# ============================================================
# 12. Initial FHM checkpoints for MUTE fine-tuning
# ============================================================

USE_INIT_CHECKPOINTS = True
FHM_CHECKPOINT_DIR = (
    SCRATCH_ROOT
    / "DL_project_alexis"
    / "checkpoints"
    / "0524-200949"
)

INIT_CHECKPOINTS = {
    "memeblip2_only": FHM_CHECKPOINT_DIR / "memeblip2_only_best.pt",
    "context_only": FHM_CHECKPOINT_DIR / "context_only_best.pt",
    "context_distillation": FHM_CHECKPOINT_DIR / "context_distillation_best.pt",
}

LOAD_INIT_STRICT = False


# ============================================================
# 13. Valid modes
# ============================================================

VALID_TRAIN_MODES = [
    "memeblip2_only",
    "context_only",
    "context_distillation",
]


# ============================================================
# 14. Pipeline force flags
# ============================================================

FORCE_RECOMPUTE_SPLITS = False
FORCE_RECOMPUTE_TEACHER = False
FORCE_RECOMPUTE_PREPROCESSING = False
FORCE_RETRAIN = True
FORCE_REEVALUATE = True


# ============================================================
# 15. Utility
# ============================================================

def ensure_project_dirs():
    for path in [
        PROJECT_ROOT,
        SPLIT_DIR,
        CACHE_DIR,
        TEACHER_OUTPUT_DIR,
        CHECKPOINT_DIR,
        LOG_DIR,
        PLOT_DIR,
        RESULTS_DIR,
        EXTERNAL_LEXICON_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def print_config_summary():
    print("================ config.py summary ================")
    print("SCRATCH_ROOT:", SCRATCH_ROOT)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("RAW_MUTE_ROOT:", RAW_MUTE_ROOT)
    print("DATASET_ROOT:", DATASET_ROOT)
    print("HF_CACHE_DIR:", HF_CACHE_DIR)

    print("SPLIT_DIR:", SPLIT_DIR)
    print("CACHE_DIR:", CACHE_DIR)
    print("TEACHER_OUTPUT_DIR:", TEACHER_OUTPUT_DIR)
    print("CHECKPOINT_DIR:", CHECKPOINT_DIR)
    print("LOG_DIR:", LOG_DIR)
    print("PLOT_DIR:", PLOT_DIR)
    print("RESULTS_DIR:", RESULTS_DIR)

    print("EXTERNAL_LEXICON_DIR:", EXTERNAL_LEXICON_DIR)
    print("HURTLEX_BN_PATH:", HURTLEX_BN_PATH)

    print("TRAIN_ROWS_PATH:", TRAIN_ROWS_PATH)
    print("VAL_ROWS_PATH:", VAL_ROWS_PATH)
    print("TEST_ROWS_PATH:", TEST_ROWS_PATH)

    print("TRAIN_TEACHER_OUTPUT_PATH:", TRAIN_TEACHER_OUTPUT_PATH)
    print("USE_EXISTING_COMMON_TEACHER_OUTPUTS:", USE_EXISTING_COMMON_TEACHER_OUTPUTS)
    print("COMMON_TEACHER_OUTPUT_PATH:", COMMON_TEACHER_OUTPUT_PATH)
    print("ALLOW_MISSING_COMMON_TEACHER_OUTPUTS:", ALLOW_MISSING_COMMON_TEACHER_OUTPUTS)

    print("VERIFYING:", VERIFYING)
    print("VERIFYING_NUM_ROWS:", VERIFYING_NUM_ROWS)
    print("FORCE_RECOMPUTE:", FORCE_RECOMPUTE)

    print("BLIP2_MODEL_NAME:", BLIP2_MODEL_NAME)
    print("TEACHER_MODEL_NAME:", TEACHER_MODEL_NAME)
    print("CLIP_MODEL_NAME:", CLIP_MODEL_NAME)
    print("CLIP_TEXT_DIM:", CLIP_TEXT_DIM)
    print("USE_CLIP_TEXT_EMBEDDINGS:", USE_CLIP_TEXT_EMBEDDINGS)

    print("SHARED_DIM:", SHARED_DIM)
    print("KNOWLEDGE_DIM:", KNOWLEDGE_DIM)
    print("KNOWLEDGE_TOP_K:", KNOWLEDGE_TOP_K)
    print("NUM_HEADS:", NUM_HEADS)
    print("USE_RESIDUAL_GATE:", USE_RESIDUAL_GATE)
    print("TRAIN_BASE:", TRAIN_BASE)
    print("FUSION_TYPE:", FUSION_TYPE)
    print("DROPOUT:", DROPOUT)
    print("PROJ_LAYERS:", PROJ_LAYERS)
    print("BETA:", BETA)
    print("TEACHER_GATE:", TEACHER_GATE)

    print("BATCH_SIZE:", BATCH_SIZE)
    print("NUM_WORKERS:", NUM_WORKERS)
    print("NUM_EPOCHS:", NUM_EPOCHS)
    print("LEARNING_RATE:", LEARNING_RATE)
    print("WEIGHT_DECAY:", WEIGHT_DECAY)
    print("MAX_GRAD_NORM:", MAX_GRAD_NORM)
    print("ETA_MIN:", ETA_MIN)
    print("LABEL_SMOOTHING:", LABEL_SMOOTHING)
    print("GRAD_ACCUM:", GRAD_ACCUM)

    print("LAMBDA_KD:", LAMBDA_KD)
    print("TEMPERATURE:", TEMPERATURE)

    print("USE_GENERICSKB:", USE_GENERICSKB)
    print("USE_EXTERNAL_LEXICONS:", USE_EXTERNAL_LEXICONS)
    print("USE_WIKIDATA:", USE_WIKIDATA)
    print("USE_WIKIPEDIA:", USE_WIKIPEDIA)
    print("TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE:", TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE)

    print("USE_INIT_CHECKPOINTS:", USE_INIT_CHECKPOINTS)
    print("FHM_CHECKPOINT_DIR:", FHM_CHECKPOINT_DIR)
    print("INIT_CHECKPOINTS:", INIT_CHECKPOINTS)
    print("LOAD_INIT_STRICT:", LOAD_INIT_STRICT)

    print("VALID_TRAIN_MODES:", VALID_TRAIN_MODES)

    print("FORCE_RECOMPUTE_SPLITS:", FORCE_RECOMPUTE_SPLITS)
    print("FORCE_RECOMPUTE_TEACHER:", FORCE_RECOMPUTE_TEACHER)
    print("FORCE_RECOMPUTE_PREPROCESSING:", FORCE_RECOMPUTE_PREPROCESSING)
    print("FORCE_RETRAIN:", FORCE_RETRAIN)
    print("FORCE_REEVALUATE:", FORCE_REEVALUATE)


if __name__ == "__main__":
    ensure_project_dirs()
    print_config_summary()
