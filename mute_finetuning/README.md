# MUTE Fine-Tuning Experiment

This folder contains the second-stage experiment for the project **Lightweight Knowledge-Grounded Detection of Hateful Memes**.

The goal of this stage is to fine-tune the same lightweight student models used in the FHM pretraining stage on the Bangla/Bengali MUTE dataset. The MUTE stage is designed as a transfer-learning experiment: the models can be initialized from the FHM-trained checkpoints and then fine-tuned on MUTE.

## Experiment Overview

This folder compares three model variants:

| Mode | Description |
|---|---|
| `memeblip2_only` | Base lightweight image-text student model |
| `context_only` | Student model with retrieved Bengali HurtLex knowledge |
| `context_distillation` | Knowledge-augmented student trained with MemeLens teacher probabilities |

The main difference from the first FHM stage is that this folder uses MUTE-specific preprocessing and Bengali/Bangla lexical knowledge. The external knowledge module intentionally loads only Bengali HurtLex from `external_lexicons_bn_only/`.

## Current Folder Structure

```text
mute_finetuning/
├── README.md
├── requirements.txt
├── config.py
├── build_mute_dataset.py
├── prepare_dataset.py
├── prepare_teacher_outputs_for_split.py
├── generate_teacher_outputs.py
├── precompute_inputs.py
├── run_mute_finetune_from_fhm.py
├── train_all_modes.py
├── evaluate.py
├── evaluate_teacher_outputs.py
├── dataset_loader.py
├── input_and_context.py
├── external_lexicon_knowledge.py
├── memeblip2.py
├── knowledge_gated.py
├── visualize_results.py
├── visualize_results_simple.py
├── external_lexicons_bn_only/
├── processed_splits/
└── teacher_outputs/
```

## Main Files

### `config.py`

Central configuration file for the MUTE fine-tuning stage.

It defines:

- project paths,
- raw MUTE dataset path,
- standardized MUTE dataset path,
- output folders,
- Bengali HurtLex path,
- teacher-output settings,
- FHM checkpoint initialization settings,
- model hyperparameters,
- training hyperparameters,
- force-recompute flags,
- verification/debug flags.


```python
PROJECT_ROOT = Path("/path/to/mute_finetuning")
RAW_MUTE_ROOT = Path("/path/to/raw_mute")
DATASET_ROOT = Path("/path/to/standardized_mute")
```

If fine-tuning from FHM checkpoints, also update `INIT_CHECKPOINTS`.

### `build_mute_dataset.py`

Converts the raw MUTE Excel files into the standardized JSONL + `images/` format used by the rest of the pipeline.

Expected raw structure:

```text
RAW_MUTE_ROOT/
└── MUTE/
    ├── train_hate.xlsx
    ├── valid_hate.xlsx
    └── test_hate.xlsx
```

Expected Excel columns:

```text
image_name
Captions
Label
```

Outputs:

```text
DATASET_ROOT/dataset.jsonl
DATASET_ROOT/dataset_unbalanced_metadata.jsonl
DATASET_ROOT/train.jsonl
DATASET_ROOT/val.jsonl
DATASET_ROOT/test.jsonl
DATASET_ROOT/images/
```

### `prepare_dataset.py`

Loads the standardized MUTE dataset from `DATASET_ROOT/dataset.jsonl`, verifies rows and images, and creates train/validation/test metadata files.

Outputs:

```text
processed_splits/train_rows.jsonl
processed_splits/val_rows.jsonl
processed_splits/test_rows.jsonl
```

### `prepare_teacher_outputs_for_split.py`

Prepares train-split teacher outputs from a common full-dataset teacher-output JSONL file.

This script is useful when teacher outputs were already generated once for the full MUTE dataset. It filters and normalizes the common teacher file so that `context_distillation` can use only the training split.

Output:

```text
teacher_outputs/train_teacher_outputs.jsonl
```

### `generate_teacher_outputs.py`

Runs the MemeLens/Qwen-VL teacher model directly on the current MUTE training split and saves soft probabilities for distillation.

Output:

```text
teacher_outputs/train_teacher_outputs.jsonl
```

### `precompute_inputs.py`

Runs the preprocessing pipeline once for train/validation/test rows and saves model-ready tensors.


- BLIP-2 image inputs,
- tokenized text,
- CLIP text embeddings,
- generated image captions,
- retrieved knowledge embeddings,
- labels and metadata.

Outputs:

```text
cached_inputs/train/
cached_inputs/val/
cached_inputs/test/
```

### `run_mute_finetune_from_fhm.py`

Main runner for the complete second-stage MUTE experiment.

It runs:

```text
1. build_mute_dataset.py
2. prepare_dataset.py
3. prepare_teacher_outputs_for_split.py or generate_teacher_outputs.py
4. precompute_inputs.py
5. train_all_modes.py --mode <mode>
6. evaluate.py --mode <mode> --checkpoint-type best
```

Recommended command:

```bash
python run_mute_finetune_from_fhm.py
```

### `train_all_modes.py`

Fine-tunes one MUTE model variant at a time.

Supported modes:

```text
memeblip2_only
context_only
context_distillation
```

This script supports loading FHM-trained checkpoints before MUTE fine-tuning. The optimizer is restarted for the MUTE stage.

Example:

```bash
python train_all_modes.py --mode context_only
```

### `evaluate.py`

Evaluates a trained MUTE checkpoint on the cached test split.

Example:

```bash
python evaluate.py --mode context_only --checkpoint-type best
```

Outputs:

```text
results/test_metrics_<mode>_best.json
results/test_predictions_<mode>_best.csv
plots/confusion_matrix_<mode>_best.png
```

### `evaluate_teacher_outputs.py`

Script for evaluating teacher predictions against MUTE training labels.

Outputs:

```text
results/teacher_metrics/teacher_metrics.json
results/teacher_metrics/teacher_predictions.csv
```

### `dataset_loader.py`

Defines the PyTorch dataset and dataloader utilities used to load cached `.pt` files during training and evaluation. It also supports loading teacher probabilities for `context_distillation`.

### `input_and_context.py`

MUTE-stage preprocessing and knowledge retrieval pipeline.

For each standardized MUTE row, it computes:

- BLIP-2 image inputs,
- BLIP-2 token IDs for compatibility/debugging,
- CLIP text embeddings,
- generated image captions,
- MiniLM knowledge embeddings from retrieved context.

This file is adapted for Bengali/Bangla and code-mixed text. It uses English spaCy mainly for English/code-mixed tokens and generated English captions, while Bengali raw OCR is handled using raw token and n-gram matching.

### `external_lexicon_knowledge.py`

Bengali/Bangla-only external lexical knowledge retrieval module.

It intentionally loads only Bengali HurtLex from:

```text
external_lexicons_bn_only/hurtlex_BN.tsv
```

It does not load the English lexicons used in the FHM stage.

The retrieved HurtLex matches are treated as contextual cues, not as labels.

### `memeblip2.py`

Defines the base MemeBLIP2-style lightweight multimodal classifier.

The model uses frozen BLIP-2 visual features and precomputed CLIP text embeddings, followed by trainable projection, adapter, fusion, and classification layers.

### `knowledge_gated.py`

Defines the context-aware MemeBLIP2 model used for:

```text
context_only
context_distillation
```

It adds cross-attention over retrieved knowledge embeddings and injects the knowledge update through an optional residual gate.

### `visualize_results.py`

Dashboard visualization script for the MUTE fine-tuning results.

Expected inputs:

```text
results/test_metrics_memeblip2_only_best.json
results/test_metrics_context_only_best.json
results/test_metrics_context_distillation_best.json

results/test_predictions_memeblip2_only_best.csv
results/test_predictions_context_only_best.csv
results/test_predictions_context_distillation_best.csv
```

Output:

```text
results_dashboard_mute.png
```

### `visualize_results_simple.py`

Visualization script that generates a bar chart comparing the three student models on accuracy, precision, recall, and F1.

Output:

```text
test_metrics_simple.png
```

## Data and Generated Folders

### `external_lexicons_bn_only/`

Contains the Bengali HurtLex file used by the external knowledge retrieval module.


Keep this folder if running `context_only` or `context_distillation`.

### `processed_splits/`

Contains processed train/validation/test metadata files generated by `prepare_dataset.py`.


### `teacher_outputs/`

Contains teacher probabilities required for `context_distillation`.


## Setup

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Running the Full MUTE Fine-Tuning Pipeline

After configuring paths in `config.py`, run:

```bash
python run_mute_finetune_from_fhm.py
```

## Running Individual Steps

### 1. Convert raw MUTE dataset

```bash
python build_mute_dataset.py
```

### 2. Prepare train/validation/test split files

```bash
python prepare_dataset.py
```

### 3. Prepare teacher outputs

If a common full-dataset teacher-output file already exists:

```bash
python prepare_teacher_outputs_for_split.py
```

Otherwise, generate teacher outputs for the train split:

```bash
python generate_teacher_outputs.py
```

### 4. Precompute model inputs

```bash
python precompute_inputs.py
```

### 5. Fine-tune models

```bash
python train_all_modes.py --mode memeblip2_only
python train_all_modes.py --mode context_only
python train_all_modes.py --mode context_distillation
```

### 6. Evaluate models

```bash
python evaluate.py --mode memeblip2_only --checkpoint-type best
python evaluate.py --mode context_only --checkpoint-type best
python evaluate.py --mode context_distillation --checkpoint-type best
```

### 7. Visualizations

```bash
python visualize_results.py
python visualize_results_simple.py --results-dir results
```

## Verification Mode

For a smaller debugging run, set in `config.py`:

```python
VERIFYING = True
```

For the full experiment, set:

```python
VERIFYING = False
```



## Notes:

- `context_distillation` requires teacher probabilities.
- The teacher model is used only during training.
- At evaluation and inference time, the student model runs independently.
- The MUTE stage can fine-tune from FHM checkpoints if `USE_INIT_CHECKPOINTS=True` in `config.py`.
- The external knowledge branch uses Bengali HurtLex cues as additional context, not as ground-truth labels.
