"""
generate_teacher_outputs.py

Run MemeLens/Qwen-VL as a teacher model on the MUTE train split and save soft
teacher probabilities for distillation.

Input:
    processed_splits/train_rows.jsonl

Output:
    teacher_outputs/train_teacher_outputs.jsonl

Each output row contains:
    id
    image_path
    text
    true_label
    source
    prob_not_hate
    prob_hate
    predicted_label
    confidence
    target
    hateful_cue
    rationale
    raw_output

Important:
    This script is resumable. If an id already exists in the output file, it is
    skipped unless FORCE_RECOMPUTE=True.

"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from config import (
    DATASET_ROOT,
    EXTERNAL_LEXICON_DIR,
    FORCE_RECOMPUTE,
    HF_CACHE_DIR,
    MAX_NEW_TOKENS,
    PROJECT_ROOT,
    TEACHER_MODEL_NAME,
    TRAIN_ROWS_PATH,
    TRAIN_TEACHER_OUTPUT_PATH,
    USE_FLASH_ATTENTION_2,
    VERIFYING,
    VERIFYING_NUM_ROWS,
    ensure_project_dirs,
)

try:
    from config import TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE as CONFIG_TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE
except Exception:
    CONFIG_TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE = True

try:
    from config import TEACHER_EXTERNAL_KNOWLEDGE_MAX_ITEMS as CONFIG_TEACHER_EXTERNAL_KNOWLEDGE_MAX_ITEMS
except Exception:
    CONFIG_TEACHER_EXTERNAL_KNOWLEDGE_MAX_ITEMS = 5

try:
    from config import TEACHER_EXTERNAL_KNOWLEDGE_MIN_SCORE as CONFIG_TEACHER_EXTERNAL_KNOWLEDGE_MIN_SCORE
except Exception:
    CONFIG_TEACHER_EXTERNAL_KNOWLEDGE_MIN_SCORE = 0.55

try:
    from external_lexicon_knowledge import retrieve_lexicon_prompt_context
    EXTERNAL_LEXICON_AVAILABLE = True
except Exception as _external_lexicon_error:
    retrieve_lexicon_prompt_context = None
    EXTERNAL_LEXICON_AVAILABLE = False
    _EXTERNAL_LEXICON_IMPORT_ERROR = _external_lexicon_error


# ============================================================
# 1. Runtime flags
# ============================================================

def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE = env_flag(
    "TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE",
    CONFIG_TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE,
)
TEACHER_EXTERNAL_KNOWLEDGE_MAX_ITEMS = int(os.getenv(
    "TEACHER_EXTERNAL_KNOWLEDGE_MAX_ITEMS",
    str(CONFIG_TEACHER_EXTERNAL_KNOWLEDGE_MAX_ITEMS),
))
TEACHER_EXTERNAL_KNOWLEDGE_MIN_SCORE = float(os.getenv(
    "TEACHER_EXTERNAL_KNOWLEDGE_MIN_SCORE",
    str(CONFIG_TEACHER_EXTERNAL_KNOWLEDGE_MIN_SCORE),
))
TEACHER_EXTERNAL_LEXICON_DIR = Path(os.getenv(
    "EXTERNAL_LEXICON_DIR",
    str(EXTERNAL_LEXICON_DIR),
))


# ============================================================
# 2. JSONL helpers
# ============================================================

def read_jsonl(path: Path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(row: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_outputs(path: Path):
    if not path.exists():
        return {}

    existing = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            existing[str(row["id"])] = row
    return existing


# ============================================================
# 3. Prompting
# ============================================================

_SYSTEM_PROMPT = (
    "You are a Bengali/Bangla and code-mixed meme classifier. "
    "Classify memes using the MUTE dataset's broad offensive-content label, "
    "not only a strict protected-group hate-speech definition. "
    "Follow the output format exactly."
)

_USER_TEMPLATE = """
Classify this Bengali/Bangla, English, or code-mixed meme.

Inputs:
- Image: provided separately.
- OCR/caption text from meme:
"{meme_text}"
- Retrieved Bengali HurtLex lexical cues: {external_knowledge_status}
{external_knowledge_block}

Label definition:
Choose Label: hateful if the meme contains clear offensive, abusive, vulgar,
humiliating, objectifying, stereotyping, derogatory, harassing, shaming, or demeaning meaning.

For this MUTE task, hateful does NOT require:
- a protected group,
- an English slur,
- an explicit threat,
- formal hate speech.

Choose Label: hateful when the image-text meaning includes one or more of these:
- sexualized or objectifying humor that reduces a person/group to body parts, sexuality, or humiliation,
- vulgar double meaning used to degrade, shame, or ridicule someone,
- gender stereotypes portraying boys, girls, women, or men as immoral, fake, inferior, dirty, greedy, or ridiculous,
- national, political, religious, or social-group mockery where a group is ridiculed or demeaned,
- Bengali slang, coded insults, or culturally implied abusive meaning,
- animal/object comparisons used to insult or dehumanize a person/group,
- visual context that makes otherwise neutral OCR offensive or degrading.

Choose Label: not-hateful when the meme is mainly:
- ordinary family, school, food, shopping, money, Eid, or relationship humor without degradation,
- sadness, loneliness, disappointment, illness, or self-deprecating humor without attacking someone,
- sports-club or fandom frustration without insulting a nationality, community, or social group,
- political commentary without direct ridicule, degradation, or abusive framing,
- ambiguous, unclear, or dependent on missing cultural context.

Bengali HurtLex cue protocol:
Use retrieved Bengali HurtLex terms as warning signals only, not labels.
A matched term only tells you what to inspect. It does not prove the meme is hateful.
Ask whether the image-text meaning actually uses the matched term as offensive, vulgar,
insulting, objectifying, stereotyping, dehumanizing, or degrading content.
If a matched term is irrelevant, quoted without endorsement, or used non-offensively, ignore it.

Calibration rules:
- Do not require explicit slurs.
- Do not label hateful just because there is a swear word; check whether it attacks, degrades, or humiliates a target.
- Do not label hateful just because someone is sad, poor, unlucky, disappointed, sick, or alone.
- If the joke is only situational and has no degrading target, choose not-hateful.
- If the meme relies on sexualized/objectifying visual humor, gender stereotyping, or social/national ridicule, prefer hateful.
- Use high confidence only when the image-text meaning is clear.
- For borderline offensive cases, use confidence 0.65-0.85 rather than 0.95+.

Output rules:
Return ONLY this exact format.
Do not write Target: hateful or Target: not-hateful.
Do not write Hateful-cue with a hyphen.
Each field must be on its own line.

Label: <hateful or not-hateful>
Confidence: <number from 0.50 to 0.99>
Target: <target person/group/category, or none/unclear>
Hateful cue: <maximum 12 words, or none>
Explanation: <one concise sentence>
"""


def get_teacher_external_knowledge(meme_text: str):
    """
    Return Bengali HurtLex cues for the teacher prompt.
    """
    if not TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE:
        return "disabled", "Not provided.\n\n", 0

    if not EXTERNAL_LEXICON_AVAILABLE:
        warning = f"External lexicon module unavailable: {_EXTERNAL_LEXICON_IMPORT_ERROR}"
        return "requested but unavailable", warning + "\n\n", 0

    result = retrieve_lexicon_prompt_context(
        meme_text=meme_text,
        image_caption="",
        lexicon_dir=TEACHER_EXTERNAL_LEXICON_DIR,
        max_items=TEACHER_EXTERNAL_KNOWLEDGE_MAX_ITEMS,
        min_score=TEACHER_EXTERNAL_KNOWLEDGE_MIN_SCORE,
        include_builtin=False,
    )

    prompt_text = result.get("prompt_text", "No Bengali HurtLex terms matched the OCR text.")
    count = int(result.get("num_candidates", 0))

    if count == 0:
        return "enabled; no Bengali HurtLex matches", prompt_text + "\n\n", count

    external_block = (
        "Retrieved Bengali HurtLex cues:\n"
        f"{prompt_text}\n\n"
        "How to use these cues:\n"
        "- Treat them as lexical warning signals only, not labels.\n"
        "- Require image-text context before choosing hateful.\n"
        "- Ignore irrelevant or non-offensive matches.\n\n"
    )
    return f"enabled; {count} Bengali HurtLex match(es)", external_block, count


def build_messages(meme_text: str):
    external_status, external_block, external_count = get_teacher_external_knowledge(meme_text)

    prompt_text = _USER_TEMPLATE.format(
        meme_text=meme_text,
        external_knowledge_status=external_status,
        external_knowledge_block=external_block,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    return messages, {
        "external_knowledge_status": external_status,
        "external_knowledge_text": external_block.strip(),
        "external_knowledge_count": external_count,
    }


# ============================================================
# 4. Parsing and validation
# ============================================================

CONFIDENCE_DEFAULT = 0.75
CONFIDENCE_MIN = 0.50
CONFIDENCE_MAX = 0.97

_FIELD_NAMES = [
    "Label",
    "Confidence",
    "Target",
    "Protected characteristic",
    "Hateful cue",
    "Explanation",
]


def extract_field(text: str, field_name: str) -> str:
    next_fields = "|".join(re.escape(name) for name in _FIELD_NAMES if name != field_name)
    pattern = (
        rf"{re.escape(field_name)}\s*:\s*"
        rf"(.*?)"
        rf"(?=\n\s*(?:{next_fields})\s*:|\Z)"
    )
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_confidence(text: str) -> float:
    confidence_text = extract_field(text, "Confidence")
    if not confidence_text:
        return CONFIDENCE_DEFAULT

    number_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", confidence_text)
    if not number_match:
        return CONFIDENCE_DEFAULT

    confidence = float(number_match.group(1))
    if confidence > 1.0:
        confidence /= 100.0

    return float(max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, confidence)))


def parse_label_output(text: str):
    label_match = re.search(
        r"Label\s*:\s*(not[-\s]?hateful|hateful)",
        text,
        re.IGNORECASE,
    )

    if not label_match:
        raise ValueError(f"Could not find Label in output: {text[:200]!r}")

    label_text = label_match.group(1).lower().replace(" ", "-")
    is_hate = label_text == "hateful"
    confidence = parse_confidence(text)

    if is_hate:
        prob_hate = confidence
        prob_not_hate = 1.0 - confidence
        predicted_label = 1
    else:
        prob_not_hate = confidence
        prob_hate = 1.0 - confidence
        predicted_label = 0

    prob_not_hate = round(float(prob_not_hate), 6)
    prob_hate = round(float(prob_hate), 6)
    confidence = round(float(confidence), 6)

    total = prob_not_hate + prob_hate
    if abs(total - 1.0) > 1e-5:
        prob_not_hate /= total
        prob_hate /= total

    target = extract_field(text, "Target") or "unclear"
    hateful_cue = extract_field(text, "Hateful cue") or "none"
    rationale = extract_field(text, "Explanation") or ""

    return {
        "prob_not_hate": prob_not_hate,
        "prob_hate": prob_hate,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "target": target,
        "protected_characteristic": "not_required_for_mute",
        "hateful_cue": hateful_cue,
        "rationale": rationale,
        "parse_failed": False,
        "parse_error": None,
    }


def fallback_teacher_result(raw_output: str, error_message: str):
    return {
        "prob_not_hate": 0.5,
        "prob_hate": 0.5,
        "predicted_label": -1,
        "confidence": 0.5,
        "target": "unclear",
        "protected_characteristic": "not_required_for_mute",
        "hateful_cue": "none",
        "rationale": "PARSE_FAILED",
        "parse_failed": True,
        "parse_error": error_message,
        "raw_output": raw_output,
    }


# ============================================================
# 5. Model loading and inference
# ============================================================

def load_teacher_model_and_processor():
    print("\n================ Loading teacher model ================")
    print("Model:", TEACHER_MODEL_NAME)
    print("HF cache:", HF_CACHE_DIR)

    processor = AutoProcessor.from_pretrained(
        TEACHER_MODEL_NAME,
        cache_dir=str(HF_CACHE_DIR),
        local_files_only=True,
    )

    # Keep image token count manageable on cluster GPUs when supported.
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
        processor.image_processor.size["longest_edge"] = 512 * 28 * 28

    model_kwargs = {
        "cache_dir": str(HF_CACHE_DIR),
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    if USE_FLASH_ATTENTION_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        TEACHER_MODEL_NAME,
        **model_kwargs,
    )
    model.eval()
    return model, processor


def get_input_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


@torch.no_grad()
def run_teacher_on_row(row: dict, model, processor):
    image_path = DATASET_ROOT / row["image_path"]
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    messages, external_prompt_metadata = build_messages(meme_text=row["text"])

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(get_input_device(model))

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

    output_text = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]

    return output_text, messages, external_prompt_metadata


# ============================================================
# 6. Main
# ============================================================

def main():
    ensure_project_dirs()

    print("================ generate_teacher_outputs.py ================")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATASET_ROOT:", DATASET_ROOT)
    print("TRAIN_ROWS_PATH:", TRAIN_ROWS_PATH)
    print("TRAIN_TEACHER_OUTPUT_PATH:", TRAIN_TEACHER_OUTPUT_PATH)
    print("TEACHER_MODEL_NAME:", TEACHER_MODEL_NAME)
    print("HF_CACHE_DIR:", HF_CACHE_DIR)
    print("VERIFYING:", VERIFYING)
    print("VERIFYING_NUM_ROWS:", VERIFYING_NUM_ROWS)
    print("FORCE_RECOMPUTE:", FORCE_RECOMPUTE)
    print("MAX_NEW_TOKENS:", MAX_NEW_TOKENS)
    print("USE_FLASH_ATTENTION_2:", USE_FLASH_ATTENTION_2)
    print("TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE:", TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE)
    print("EXTERNAL_LEXICON_DIR:", TEACHER_EXTERNAL_LEXICON_DIR)
    print("TEACHER_EXTERNAL_KNOWLEDGE_MAX_ITEMS:", TEACHER_EXTERNAL_KNOWLEDGE_MAX_ITEMS)
    print("TEACHER_EXTERNAL_KNOWLEDGE_MIN_SCORE:", TEACHER_EXTERNAL_KNOWLEDGE_MIN_SCORE)

    if not TRAIN_ROWS_PATH.exists():
        raise FileNotFoundError(
            f"Train split not found: {TRAIN_ROWS_PATH}\n"
            "Run prepare_dataset.py first."
        )

    rows = read_jsonl(TRAIN_ROWS_PATH)
    print(f"\n[Info] Loaded {len(rows)} train rows.")

    if VERIFYING:
        rows = rows[:VERIFYING_NUM_ROWS]
        print(f"[Verification] Using only first {len(rows)} row(s).")

    existing_outputs = load_existing_outputs(TRAIN_TEACHER_OUTPUT_PATH)
    print(f"[Info] Existing teacher outputs: {len(existing_outputs)}")

    model, processor = load_teacher_model_and_processor()

    processed = 0
    skipped = 0
    failed = 0

    for row in tqdm(rows, desc="Generating teacher outputs"):
        row_id = str(row["id"])

        if row_id in existing_outputs and not FORCE_RECOMPUTE:
            existing_variant = bool(existing_outputs[row_id].get(
                "teacher_external_knowledge_included",
                False,
            ))
            if existing_variant == bool(TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE):
                skipped += 1
                continue

        try:
            raw_output, messages, external_prompt_metadata = run_teacher_on_row(
                row=row,
                model=model,
                processor=processor,
            )

            try:
                normalized = parse_label_output(raw_output)
                normalized["raw_output"] = raw_output
            except Exception as parse_error:
                normalized = fallback_teacher_result(
                    raw_output=raw_output,
                    error_message=str(parse_error),
                )

            output_row = {
                "id": row_id,
                "image_path": row["image_path"],
                "text": row["text"],
                "true_label": int(row["label"]),
                "source": row.get("source", "mute"),
                "teacher_external_knowledge_included": bool(TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE),
                "teacher_external_knowledge_status": external_prompt_metadata.get("external_knowledge_status"),
                "teacher_external_knowledge_count": int(external_prompt_metadata.get("external_knowledge_count", 0)),
                "teacher_external_knowledge_text": external_prompt_metadata.get("external_knowledge_text", ""),
                **normalized,
            }

            # Backward-compatible aliases for older evaluation/training scripts.
            output_row["label"] = output_row["true_label"]
            output_row["teacher_pred_label"] = output_row["predicted_label"]
            output_row["teacher_probs"] = [output_row["prob_not_hate"], output_row["prob_hate"]]
            output_row["teacher_response"] = output_row["raw_output"]

            append_jsonl(output_row, TRAIN_TEACHER_OUTPUT_PATH)
            processed += 1

            if VERIFYING:
                print("\n================ Verification sample ================")
                print("id:", row_id)
                print("image:", DATASET_ROOT / row["image_path"])
                print("text:", row["text"])
                print("true_label:", row["label"])
                print("\nPrompt text:")
                print(messages[1]["content"][1]["text"])
                print("\nRaw model output:")
                print(raw_output)
                print("\nParsed/saved output:")
                print(json.dumps(output_row, indent=2, ensure_ascii=False))

        except Exception as error:
            failed += 1
            error_row = {
                "id": row_id,
                "image_path": row.get("image_path"),
                "text": row.get("text"),
                "true_label": int(row.get("label", -1)),
                "label": int(row.get("label", -1)),
                "source": row.get("source", "mute"),
                "teacher_external_knowledge_included": bool(TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE),
                "teacher_external_knowledge_status": "error before/during teacher call",
                "teacher_external_knowledge_count": 0,
                "teacher_external_knowledge_text": "",
                "prob_not_hate": 0.5,
                "prob_hate": 0.5,
                "teacher_probs": [0.5, 0.5],
                "predicted_label": -1,
                "teacher_pred_label": -1,
                "confidence": 0.5,
                "target": "unclear",
                "protected_characteristic": "not_required_for_mute",
                "hateful_cue": "none",
                "rationale": "TEACHER_INFERENCE_FAILED",
                "parse_failed": True,
                "parse_error": str(error),
                "raw_output": "",
                "teacher_response": "",
            }
            append_jsonl(error_row, TRAIN_TEACHER_OUTPUT_PATH)
            print(f"\n[Warning] Failed row id={row_id}: {error}")

    print("\n================ Summary ================")
    print("Processed:", processed)
    print("Skipped existing:", skipped)
    print("Failed:", failed)
    print("Output file:", TRAIN_TEACHER_OUTPUT_PATH)
    print("\n[Done] Teacher outputs generated.")


if __name__ == "__main__":
    main()
