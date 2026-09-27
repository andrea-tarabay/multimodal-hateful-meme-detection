#!/usr/bin/env bash
set -euo pipefail

cd /scratch/DL_project_alexis

echo "===== Repo/status ====="
git config --global --add safe.directory /scratch/DL_project_alexis || true
git status

echo "===== Compile check ====="
python3 -m py_compile \
  config.py \
  prepare_dataset.py \
  generate_teacher_outputs.py \
  precompute_inputs.py \
  train_all_modes.py \
  evaluate.py \
  run_pipeline.py

echo "===== Clean old runtime outputs ====="
rm -rf processed_splits cached_inputs teacher_outputs checkpoints logs plots results

echo "===== Configure full FHM dataset run ====="
python3 - <<'PY'
from pathlib import Path
import re

p = Path("config.py")
s = p.read_text()

def set_line(name, value):
    global s
    if re.search(rf"^{name}\s*=", s, flags=re.M):
        s = re.sub(rf"^{name}\s*=.*", f"{name} = {value}", s, flags=re.M)
    else:
        s += f"\n{name} = {value}\n"

s = re.sub(
    r'PROJECT_ROOT\s*=\s*SCRATCH_ROOT\s*/\s*["\'].*?["\']',
    'PROJECT_ROOT = SCRATCH_ROOT / "DL_project_alexis"',
    s,
)

s = re.sub(
    r'DATASET_ROOT\s*=.*',
    'DATASET_ROOT = SCRATCH_ROOT / "meme_dataset_separated" / "fhm"',
    s,
)

set_line("VERIFYING", "False")
set_line("BALANCE_CLASSES", "True")
set_line("FORCE_RECOMPUTE", "True")
set_line("FORCE_RECOMPUTE_SPLITS", "True")
set_line("FORCE_RECOMPUTE_TEACHER", "True")
set_line("FORCE_RECOMPUTE_PREPROCESSING", "True")
set_line("FORCE_RETRAIN", "True")
set_line("FORCE_REEVALUATE", "True")
set_line("TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE", "True")

p.write_text(s)
print("Configured full FHM dataset run.")
PY

echo "===== Run full pipeline on full FHM dataset ====="
USE_EXTERNAL_LEXICONS=1 TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE=1 python3 run_pipeline.py

echo "===== Final run finished ====="
