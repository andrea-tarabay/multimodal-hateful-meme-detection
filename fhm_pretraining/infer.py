"""
infer.py

Run full inference pipeline on a single meme.

Steps:
    1. Load and preprocess the image (BLIP-2 pixel values + BLIP caption).
    2. Compute the pretrained CLIP text embedding.
    3. Retrieve external knowledge contexts (GenericsKB ).
    4. Load the trained model from a checkpoint.
    5. Run forward pass and print the hate-meme prediction.

NOTE: WE ONLY USE DATA DROM GENERICSKB FOR THE FINAL RESULTS, BECAUSE WIKIDATA AND WIKIPEDIA RETRIEVAL HAD LOW QUALITY AND DID NOT IMPROVE THE MODEL. 

    
Usage:
    python infer.py --image path/to/meme.jpg --text "meme text here"
    python infer.py --image path/to/meme.jpg --text "text" --mode context_only
    python infer.py --image path/to/meme.jpg --text "text" --mode memeblip2_only --checkpoint-type last
    python infer.py --image path/to/meme.jpg --text "text" --run-name 0524-200949 --verbose
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ── ensure we import from the right project directory ──────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import (
    BLIP2_MODEL_NAME,
    NUM_CLASSES,
    SHARED_DIM,
    KNOWLEDGE_DIM,
    NUM_HEADS,
    USE_RESIDUAL_GATE,
    TRAIN_BASE,
    CLIP_TEXT_DIM,
    VALID_TRAIN_MODES,
)

from memeblip2 import MemeBLIP2
from knowledge_gated import CrossAttentionKnowledgeMemeBLIP2

# Suppress the "UNEXPECTED keys" table that transformers prints when loading
# CLIPTextModel from the full CLIP checkpoint (vision weights are intentionally unused).
from transformers.utils import logging as _hf_logging
_hf_logging.set_verbosity_error()

# input_and_context loads heavy models at import time (caption model, CLIP, spacy…)
print("[infer] Loading preprocessing models (caption, CLIP, spacy, GenericsKB)…", flush=True)
from input_and_context import (
    process_meme_sample,
    retrieve_three_contexts_per_query_term,
    generics_df,
    USE_WIKIDATA, WIKIDATA_RESULTS_PER_TERM, USE_WIKIDATA_EXACT_LABEL_FILTER,
    USE_WIKIPEDIA, WIKIPEDIA_RESULTS_PER_TERM, WIKIPEDIA_SENTENCES_PER_PAGE,
    WIKIPEDIA_EXACT_TITLE_MATCH_ONLY, WIKIPEDIA_MAX_CONTEXT_WORDS, WIKIPEDIA_LANGUAGE,
    USE_GENERICSKB, GENERICS_RESULTS_PER_TERM, MIN_GENERICS_SCORE,
    ALPHA_BM25, BETA_EMBEDDING, GAMMA_SOURCE,
    KNOWLEDGE_TOP_K, NORMALIZE_KNOWLEDGE_EMBEDDINGS, MAX_QUERY_TERMS,
    MAX_TEXT_LENGTH,
)

try:
    from input_and_context import (
        USE_EXTERNAL_LEXICONS,
        external_lexicon_df,
        EXTERNAL_LEXICON_RESULTS_PER_TERM,
        EXTERNAL_LEXICON_MIN_SCORE,
    )
    _HAS_EXTERNAL_LEXICONS = True
except ImportError:
    _HAS_EXTERNAL_LEXICONS = False


# ── Model builders ─────────────────────────────────────────────────────────────

def _infer_arch_from_state_dict(sd: dict, mode: str) -> dict:
    """Infer model constructor kwargs from a checkpoint's state_dict shapes."""
    prefix = "base." if mode in ("context_only", "context_distillation") else ""

    img_proj_w = sd[f"{prefix}image_projection.fc.0.weight"]
    shared_dim = img_proj_w.shape[0]

    txt_proj_w = sd[f"{prefix}text_projection.fc.0.weight"]
    clip_text_dim = txt_proj_w.shape[1]

    proj_layers = sum(
        1 for k in sd if k.startswith(f"{prefix}image_projection.fc.") and k.endswith(".weight")
    )

    pre_out_w = sd[f"{prefix}pre_output.0.weight"]
    fusion_type = "concat" if pre_out_w.shape[1] == 2 * shared_dim else "multiply"

    num_classes = sd[f"{prefix}classifier.3.weight"].shape[0]

    arch = dict(
        shared_dim=shared_dim,
        clip_text_dim=clip_text_dim,
        proj_layers=proj_layers,
        fusion_type=fusion_type,
        num_classes=num_classes,
    )

    if mode in ("context_only", "context_distillation"):
        kp_w = sd["knowledge_projection.fc.0.weight"]
        arch["knowledge_dim"] = kp_w.shape[1]
        arch["use_residual_gate"] = "residual_gate.0.weight" in sd

        # num_heads: pick the largest power-of-2 divisor of shared_dim that
        # keeps head_dim >= 32 (standard safe choice).
        for h in [8, 4, 2, 1]:
            if shared_dim % h == 0 and shared_dim // h >= 32:
                arch["num_heads"] = h
                break

    return arch


def build_model_from_checkpoint(checkpoint_path: Path, mode: str, device: str):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    sd = checkpoint["model_state_dict"]

    arch = _infer_arch_from_state_dict(sd, mode)
    print(f"[infer] Inferred architecture: {arch}")

    if mode == "memeblip2_only":
        model = MemeBLIP2(
            blip2_name=BLIP2_MODEL_NAME,
            **arch,
        )
    else:
        model = CrossAttentionKnowledgeMemeBLIP2(
            blip2_name=BLIP2_MODEL_NAME,
            train_base=TRAIN_BASE,
            **arch,
        )

    model.load_state_dict(sd)
    model = model.to(device)

    print(f"[infer] Loaded checkpoint: {checkpoint_path}")
    if checkpoint.get("epoch") is not None:
        print(f"        epoch={checkpoint['epoch']}  mode={checkpoint.get('mode')}  "
              f"metrics={checkpoint.get('metrics')}")
    return model


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_meme(image_path: str, meme_text: str, mode: str, verbose: bool):
    """Run the full preprocessing pipeline for a single meme."""

    print(f"[infer] Preprocessing image: {image_path}", flush=True)
    sample = process_meme_sample(
        image_path=image_path,
        meme_text=meme_text,
        max_text_length=MAX_TEXT_LENGTH,
    )

    print(f"[infer] Image caption: {sample['image_caption']!r}")

    knowledge_embeddings = torch.zeros(KNOWLEDGE_TOP_K, KNOWLEDGE_DIM, dtype=torch.float32)
    knowledge_mask = torch.zeros(KNOWLEDGE_TOP_K, dtype=torch.bool)

    if mode in ("context_only", "context_distillation"):
        print("[infer] Retrieving external knowledge…", flush=True)

        retrieval_kwargs = dict(
            meme_text=sample["ocr_text"],
            image_caption=sample["image_caption"],
            generics_df=generics_df,
            max_terms=MAX_QUERY_TERMS,
            use_wikidata=USE_WIKIDATA,
            wikidata_results_per_term=WIKIDATA_RESULTS_PER_TERM,
            use_wikidata_exact_label_filter=USE_WIKIDATA_EXACT_LABEL_FILTER,
            use_wikipedia=USE_WIKIPEDIA,
            wikipedia_results_per_term=WIKIPEDIA_RESULTS_PER_TERM,
            wikipedia_sentences_per_page=WIKIPEDIA_SENTENCES_PER_PAGE,
            wikipedia_exact_title_match_only=WIKIPEDIA_EXACT_TITLE_MATCH_ONLY,
            wikipedia_max_context_words=WIKIPEDIA_MAX_CONTEXT_WORDS,
            wikipedia_language=WIKIPEDIA_LANGUAGE,
            use_genericskb=USE_GENERICSKB,
            generics_results_per_term=GENERICS_RESULTS_PER_TERM,
            min_generics_score=MIN_GENERICS_SCORE,
            alpha_bm25=ALPHA_BM25,
            beta_embedding=BETA_EMBEDDING,
            gamma_source=GAMMA_SOURCE,
            knowledge_top_k=KNOWLEDGE_TOP_K,
            normalize_knowledge_embeddings=NORMALIZE_KNOWLEDGE_EMBEDDINGS,
            verbose=verbose,
        )

        if _HAS_EXTERNAL_LEXICONS:
            retrieval_kwargs.update(
                use_external_lexicons=USE_EXTERNAL_LEXICONS,
                external_lexicon_df=external_lexicon_df,
                external_lexicon_results_per_term=EXTERNAL_LEXICON_RESULTS_PER_TERM,
                external_lexicon_min_score=EXTERNAL_LEXICON_MIN_SCORE,
            )

        retrieval_result = retrieve_three_contexts_per_query_term(**retrieval_kwargs)

        knowledge_embeddings = torch.tensor(
            retrieval_result["knowledge_embeddings"], dtype=torch.float32
        )
        knowledge_mask = torch.tensor(
            retrieval_result["knowledge_mask"], dtype=torch.bool
        )

        if verbose:
            print(f"[infer] knowledge_mask: {knowledge_mask.tolist()}")
            selected_df = retrieval_result.get("selected_contexts_df")
            if selected_df is not None and not selected_df.empty:
                print("[infer] Top knowledge contexts:")
                for _, row in selected_df.iterrows():
                    if row.get("context_mask", 0) == 1 and row.get("sentence"):
                        print(f"  - [{row.get('source', '?')}] {row['sentence']}")

    # pixel_values from process_meme_sample has shape [1, 3, H, W] — squeeze batch dim
    pixel_values = sample["pixel_values"].squeeze(0).float()
    clip_text_embedding = sample["clip_text_embedding"].squeeze(0).float()

    return pixel_values, clip_text_embedding, knowledge_embeddings, knowledge_mask


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, pixel_values, clip_text_embedding, knowledge_embeddings,
                  knowledge_mask, mode: str, device: str):
    model.eval()

    # Add batch dimension
    pixel_values = pixel_values.unsqueeze(0).to(device)
    clip_text_embedding = clip_text_embedding.unsqueeze(0).to(device)
    knowledge_embeddings = knowledge_embeddings.unsqueeze(0).to(device)
    knowledge_mask = knowledge_mask.unsqueeze(0).to(device)

    if mode == "memeblip2_only":
        logits = model(
            pixel_values=pixel_values,
            clip_text_embedding=clip_text_embedding,
        )
    else:
        outputs = model(
            pixel_values=pixel_values,
            clip_text_embedding=clip_text_embedding,
            knowledge_embeddings=knowledge_embeddings,
            knowledge_mask=knowledge_mask,
            return_dict=True,
        )
        logits = outputs["logits"]

    probs = F.softmax(logits, dim=-1).squeeze(0)
    pred = torch.argmax(probs).item()

    return pred, probs.cpu().tolist()


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hate-meme inference on a single image+text pair."
    )
    parser.add_argument("--image", required=True,
                        help="Path to the meme image file.")
    parser.add_argument("--text", required=True,
                        help="Text written on / associated with the meme.")
    parser.add_argument("--mode", default="context_only",
                        choices=["memeblip2_only", "context_only", "context_distillation"],
                        help="Which trained model to use (default: context_only).")
    parser.add_argument("--checkpoint-type", default="best", choices=["best", "last"],
                        help="Use the 'best' or 'last' saved checkpoint (default: best).")
    parser.add_argument("--run-name", default="0524-200949",
                        help="Run name subfolder under checkpoints/ (default: 0524-200949).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print retrieved knowledge contexts and extra debug info.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even when CUDA is available.")
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"[infer] device={device}  mode={args.mode}  run={args.run_name}  "
          f"checkpoint={args.checkpoint_type}")

    checkpoint_path = (
        PROJECT_DIR / "checkpoints" / args.run_name
        / f"{args.mode}_{args.checkpoint_type}.pt"
    )

    # ── preprocessing ─────────────────────────────────────────────────────────
    pixel_values, clip_text_embedding, knowledge_embeddings, knowledge_mask = (
        preprocess_meme(args.image, args.text, args.mode, args.verbose)
    )

    # ── model ─────────────────────────────────────────────────────────────────
    print("[infer] Building model…", flush=True)
    model = build_model_from_checkpoint(checkpoint_path, args.mode, device)

    # ── forward pass ──────────────────────────────────────────────────────────
    print("[infer] Running forward pass…", flush=True)
    pred, probs = run_inference(
        model, pixel_values, clip_text_embedding,
        knowledge_embeddings, knowledge_mask, args.mode, device,
    )

    label_name = "HATEFUL" if pred == 1 else "NOT HATEFUL"
    print("\n" + "=" * 60)
    print(f"  Prediction : {label_name}  (class {pred})")
    print(f"  P(not hate): {probs[0]:.4f}")
    print(f"  P(hate)    : {probs[1]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
