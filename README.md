# Lightweight Knowledge-Grounded Detection of Hateful Memes

This repository contains the code for the master’s project **Lightweight Knowledge-Grounded Detection of Hateful Memes**.

The project studies whether a lightweight multimodal hateful meme detector can be improved using external knowledge retrieval and teacher distillation. The main motivation is that hateful memes often require understanding the interaction between image, text, humor, stereotypes, symbols, and cultural context. Large vision-language models can provide stronger contextual reasoning, but they are expensive to store and deploy. Therefore, this project investigates whether a smaller MemeBLIP2-style student model can remain deployable while benefiting from additional contextual information.

## Project Summary

We compare three student model variants:

| Variant | Description |
|---|---|
| `memeblip2_only` | Base lightweight image-text model |
| `context_only` | Adds external hate/offensive knowledge retrieval |
| `context_distillation` | Adds MemeLens teacher distillation during training |

The models are trained in two stages:

1. **FHM pretraining**: training and evaluation on the English Facebook Hateful Memes dataset.
2. **MUTE fine-tuning**: transfer fine-tuning on the Bangla MUTE dataset.

The first stage tests the model on English hateful memes. The second stage evaluates whether the FHM-trained checkpoints can transfer to a multilingual setting where Bangla-specific offensive cues may be useful.

## Repository Structure

```text
knowledge_grounded_hateful_memes/
├── README.md
├── fhm_pretraining/
├── mute_finetuning/
├── docker_image/
├── dataset_related/
├── code_running_video.mp4/
├── Stage1_FHM_Results_Dashboard.jpeg/
├── Stage2_MUTE_Results_Dashboard.jpeg/
├── EE_559__Group03_Poster.pdf
└── EE_559__Group03_Report.pdf
```

## Stages: Folders Descriptions 

### `fhm_pretraining/`

This folder contains the first-stage experiment on the Facebook Hateful Memes dataset.

It includes the full pipeline for:

- preparing FHM splits,
- generating MemeLens teacher outputs,
- precomputing image, text, caption, and knowledge features,
- training the three model variants,
- evaluating the trained checkpoints on the FHM test set.

The FHM-trained checkpoints are then used as initialization for the second-stage MUTE fine-tuning experiment.

A README inside this folder details scripts functionalities and how to run them.

### `mute_finetuning/`

This folder contains the second-stage experiment on the Bangla/Bengali MUTE dataset.

It includes the full pipeline for:

- converting the raw MUTE Excel files into the standardized JSONL/image format,
- preparing train/validation/test metadata,
- preparing or generating MemeLens teacher outputs,
- precomputing image, text, caption, and Bengali HurtLex knowledge features,
- fine-tuning the three student model variants from FHM checkpoints,
- evaluating and visualizing the MUTE results.

A README inside this folder details scripts functionalities and how to run them.
