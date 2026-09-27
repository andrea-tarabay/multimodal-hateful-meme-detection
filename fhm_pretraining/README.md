# FHM Pretraining Experiment

This folder contains the first-stage experiment for the project **Lightweight Knowledge-Grounded Detection of Hateful Memes**.

The goal of this experiment is to train and evaluate lightweight hateful meme detection models on the English Facebook Hateful Memes (FHM) dataset. The trained checkpoints from this stage are then used to initialize the second-stage MUTE fine-tuning experiment.

## Experiment Overview

The experiment compares three student model variants:

| Mode | Description |
|---|---|
| `memeblip2_only` | Base lightweight image-text student model |
| `context_only` | Student model with retrieved external knowledge |
| `context_distillation` | Knowledge-augmented student trained with MemeLens teacher probabilities |

The base model follows a MemeBLIP2-style architecture. Large pretrained encoders are frozen, while lightweight projection, adapter, fusion, cross-attention, and classification layers are trained.

## Folder Structure

```text
fhm_pretraining/
├── README.md
├── requirements.txt
├── config.py
├── prepare_dataset.py
├── run_pipeline.py
├── train_all_modes.py
├── evaluate.py
├── evaluate_teacher_outputs.py
├── generate_teacher_outputs.py
├── precompute_inputs.py
├── dataset_loader.py
├── input_and_context.py
├── external_lexicon_knowledge.py
├── build_external_lexicons_combined.py
├── infer.py
├── memeblip2.py
├── knowledge_gated.py
├── dataset_related/
├── external_lexicons/
├── scripts/
├── processed_splits/
└── teacher_outputs/
```

## Main Files

### `config.py`

Central configuration file containing dataset paths, output folders, model settings, training hyperparameters, distillation parameters, and debug flags.

Before running the experiment, update the paths:

```python
PROJECT_ROOT = Path("/path/to/fhm_pretraining")
DATASET_ROOT = Path("/path/to/fhm")
```

### `prepare_dataset.py`

Loads the raw FHM dataset, verifies rows and image paths, and creates train/validation/test metadata files.

Outputs:

```text
processed_splits/train_rows.jsonl
processed_splits/val_rows.jsonl
processed_splits/test_rows.jsonl
```

### `run_pipeline.py`

Runs the complete experiment pipeline end-to-end:

```text
1. prepare_dataset.py
2. generate_teacher_outputs.py
3. precompute_inputs.py
4. train_all_modes.py --mode memeblip2_only
5. train_all_modes.py --mode context_only
6. train_all_modes.py --mode context_distillation
7. evaluate.py --mode memeblip2_only
8. evaluate.py --mode context_only
9. evaluate.py --mode context_distillation
```

Recommended command:

```bash
python run_pipeline.py
```

### `generate_teacher_outputs.py`

Runs the MemeLens teacher model on the FHM training split and saves soft probabilities for distillation.

Output:

```text
teacher_outputs/train_teacher_outputs.jsonl
```

This file is required for the `context_distillation` mode.

### `precompute_inputs.py`

Precomputes model-ready tensors so that expensive preprocessing is not repeated during every training epoch.

It caches:

- BLIP-2 image inputs,
- tokenized meme text,
- CLIP text embeddings,
- generated image captions,
- retrieved external knowledge embeddings,
- labels and metadata.

Outputs:

```text
cached_inputs/train/
cached_inputs/val/
cached_inputs/test/
```

### `train_all_modes.py`

Trains one model variant at a time.

Supported modes:

```text
memeblip2_only
context_only
context_distillation
```

Example:

```bash
python train_all_modes.py --mode context_only
```

Outputs:

```text
checkpoints/
logs/
plots/
```

### `evaluate.py`

Evaluates a saved checkpoint on the cached FHM test split.

Example:

```bash
python evaluate.py --mode context_only
```

Outputs:

```text
results/
```

### `evaluate_teacher_outputs.py`

Script used to evaluate MemeLens teacher predictions against the dataset labels. This is for checking teacher quality or comparing teacher prompting strategies.

### `dataset_loader.py`

Defines the PyTorch dataset and dataloader utilities used to load cached `.pt` files during training and evaluation. It also attaches teacher probabilities when running `context_distillation`.

### `input_and_context.py`

Main preprocessing and knowledge retrieval module. It generates image captions, computes CLIP text embeddings, retrieves external knowledge from enabled sources, and returns tensors compatible with the training pipeline.

### `external_lexicon_knowledge.py`

Loads and retrieves local hate/offensive lexical knowledge. The retrieved lexicon entries are used as contextual cues, not as labels.

### `build_external_lexicons_combined.py`

Optional helper script for combining separate external lexicon files into one CSV. The main pipeline can also load the separate lexicon files directly.

### `infer.py`

Optional script for running inference on a single meme using a trained checkpoint.

Example:

```bash
python infer.py --image path/to/meme.jpg --text "meme text here" --mode context_only
```

### `memeblip2.py`

Defines the base lightweight multimodal classifier. It uses frozen BLIP-2 visual features and precomputed CLIP text embeddings, followed by trainable projection, adapter, fusion, and classification layers.

### `knowledge_gated.py`

Defines the knowledge-augmented model used for `context_only` and `context_distillation`. It adds cross-attention over retrieved knowledge embeddings and injects the knowledge update through a residual gate.

## Folders

### `dataset_related/`

Contains dataset preparation scripts. 

### `external_lexicons/`

Contains external hate/offensive lexicon files used by `external_lexicon_knowledge.py`.

Keep this folder if running the context-based models.

### `scripts/`

Contains auxiliary scripts used during experimentation or cluster execution. These are not required if running the pipeline directly with `run_pipeline.py`.

### `processed_splits/`

Contains processed train/validation/test metadata files generated by `prepare_dataset.py`.

### `teacher_outputs/`

Contains teacher probabilities generated by `generate_teacher_outputs.py`.


## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Expected FHM Dataset Format

The dataset folder should contain:

```text
train.jsonl
val.jsonl
test.jsonl
images/
```

Each row should include:

```text
id
image_path
text
label
source
```

## Running the Full Experiment

After updating paths in `config.py`, run:

```bash
python run_pipeline.py
```

## Running Individual Steps

```bash
python prepare_dataset.py
python generate_teacher_outputs.py
python precompute_inputs.py

python train_all_modes.py --mode memeblip2_only
python train_all_modes.py --mode context_only
python train_all_modes.py --mode context_distillation

python evaluate.py --mode memeblip2_only
python evaluate.py --mode context_only
python evaluate.py --mode context_distillation
```

## Verification Mode

For a small debugging run, set in `config.py`:

```python
VERIFYING = True
```

For the full experiment, set:

```python
VERIFYING = False
```


## Notes

- `context_distillation` requires teacher probabilities from `teacher_outputs/train_teacher_outputs.jsonl`.
- The teacher model is used only during training.
- At inference time, the student model runs independently.
- The FHM checkpoints from this folder are used to initialize the second-stage MUTE fine-tuning experiment.
