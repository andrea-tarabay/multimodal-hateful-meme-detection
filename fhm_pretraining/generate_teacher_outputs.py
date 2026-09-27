"""
generate_teacher_outputs.py

Purpose:
    Run QCRI/MemeLens-VLM as a teacher model on the TRAIN split
    and save teacher probabilities for distillation.

This script should be run once after prepare_dataset.py.

Input:
    processed_splits/train_rows.jsonl

Output:
    teacher_outputs/train_teacher_outputs.jsonl

Each output row contains:
    id
    prob_not_hate
    prob_hate
    predicted_label
    confidence
    rationale
    raw_output

Important:
    This script is resumable. If an id already exists in the output file,
    it is skipped unless FORCE_RECOMPUTE=True.

VERIFYING mode:
    Controlled from config.py.
    If VERIFYING=True, only one/few rows are processed and detailed checks are printed.
"""

import json
import re
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from config import (
    DATASET_ROOT,
    PROJECT_ROOT,
    HF_CACHE_DIR,
    TRAIN_ROWS_PATH,
    TRAIN_TEACHER_OUTPUT_PATH,
    TEACHER_MODEL_NAME,
    VERIFYING,
    VERIFYING_NUM_ROWS,
    FORCE_RECOMPUTE,
    MAX_NEW_TOKENS,
    USE_FLASH_ATTENTION_2,
    ensure_project_dirs,
)

# Optional config values for teacher prompt ablation.
try:
    from config import TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE as CONFIG_TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE
except Exception:
    CONFIG_TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE = False

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
EXTERNAL_LEXICON_DIR = Path(os.getenv(
    "EXTERNAL_LEXICON_DIR",
    str(PROJECT_ROOT / "external_lexicons"),
))


# ============================================================
# 1. JSONL helpers
# ============================================================

def read_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
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
            existing[row["id"]] = row

    return existing


# ============================================================
# 2. Prompting
# ============================================================

_SYSTEM_PROMPT = (
    "You are an expert social media image analyzer specializing in "
    "identifying hateful content in memes."
)

_USER_TEMPLATE = """
Analyze this meme for hateful content.

Inputs:
- Image: provided separately.
- OCR text from meme: {meme_text}
- Retrieved external knowledge cues: {external_knowledge_status}
{external_knowledge_block}

Definition:
Classify as hateful if the meme attacks, demeans, dehumanizes, stereotypes, threatens, excludes,
or expresses contempt toward a protected group or a person because of protected characteristics such as
race, ethnicity, nationality, religion, gender, sex, sexual orientation, disability, or immigration status.

Classify as not-hateful if the meme is neutral, political without protected-group hate, counters hate,
quotes hateful language without endorsing it, is ambiguous without enough evidence, or targets a non-protected
category such as a job, political party, public figure, fictional character, or general behavior.

Important distinction:
Offensive, rude, crude, dark, shocking, or violent humor is NOT automatically hateful.
It becomes hateful only when the harmful meaning is directed at a protected group or at a person because of protected identity.

External knowledge use protocol:
Use retrieved external cues as a checklist, not as labels.
A cue only tells you what to inspect. It does not prove the meme is hateful.
For each cue, ask:
1. Does the cue identify a possible protected-group target?
2. Is the group actually being attacked, stereotyped, mocked, dehumanized, threatened, excluded, or framed as inferior/dangerous?
3. Is the negative meaning applied because of protected identity, rather than because of an individual action or unrelated joke?
4. If the cue is generic, weak, or only an identity mention, ignore it.

Implicit hate rule:
The target does not need to be stated explicitly.
A protected target can be implied by:
- names associated with religion, ethnicity, nationality, gender, or culture,
- visual symbols, clothing, skin color, disability aids, flags, religious signs, or stereotypes,
- coded associations with crime, violence, dirtiness, inferiority, animals, disease, sexuality, or exclusion,
- sarcasm or joke framing where the joke depends on degrading a protected group.

But be careful:
A protected identity word by itself is not hateful.
A meme that merely says someone is Jewish, Muslim, Christian, Black, white, gay, disabled, immigrant, etc. is not hateful unless there is a negative claim, stereotype, contempt, threat, dehumanization, or exclusion.

Decision checklist:
1. Read the OCR text literally.
2. Inspect the image for people, symbols, gestures, objects, and visual stereotypes.
3. Identify the target, including implied targets.
4. Decide whether the target is a protected group or a person targeted because of protected identity.
5. Decide whether the meme expresses hateful meaning, including implicit, sarcastic, coded, or stereotype-based meaning.
6. Check whether external cues support a real harmful relation, not just an identity mention.
7. If evidence is weak, unclear, or only offensive without protected-group targeting, choose not-hateful with lower confidence.

Consistency rules:
- If Target is none/unclear AND Hateful cue is none, the label should be not-hateful.
- If the meme has a protected target AND a negative stereotype, contempt, dehumanization, threat, exclusion, or inferiority claim, the label should be hateful, even if phrased as a joke.
- If the explanation says there is no protected target or no hateful cue, do not output Label: hateful.
- The final label must match the Target, Protected characteristic, Hateful cue, and Explanation fields.

Examples of the required output format:
Example A:
OCR text: protected group X are all criminals
Label: hateful
Confidence: 0.90
Target: protected group X
Protected characteristic: race/ethnicity
Hateful cue: criminality stereotype
Explanation: The meme links a protected group to criminality, which is a harmful stereotype.

Example B:
OCR text: politicians are clowns
Label: not-hateful
Confidence: 0.88
Target: politicians
Protected characteristic: none
Hateful cue: none
Explanation: The meme insults a political category, not a protected group.

Example C:
OCR text: character Y is Jewish
Label: not-hateful
Confidence: 0.82
Target: character Y
Protected characteristic: religion
Hateful cue: none
Explanation: The text mentions a religious identity but does not attack, stereotype, threaten, or demean that group.

Example D:
OCR text: group X are dirty animals
Label: hateful
Confidence: 0.92
Target: group X
Protected characteristic: unclear protected identity
Hateful cue: dehumanizing comparison
Explanation: The meme dehumanizes a group by comparing them to animals, which is a hateful cue if the group is protected or identity-coded.

Return ONLY the following format.
Do not copy the placeholder text. Do not write [SPECIFY ...].
If a field is unknown, write exactly 'unclear' or 'none'.
Each field must be on its own separate line.
Confidence must be written exactly as 'Confidence:', not any other spelling.
Use confidence carefully: 0.50-0.65 for ambiguous cases, 0.66-0.85 for moderate evidence,
and above 0.85 only when the hateful or non-hateful signal is very clear.

Label: <hateful or not-hateful>
Confidence: <number from 0.50 to 0.99>
Target: <target group/person, or none/unclear>
Protected characteristic: <race/religion/gender/etc., or none/unclear>
Hateful cue: <maximum 12 words, or none>
Explanation: <one or two concise sentences explaining the decision>
"""



def get_teacher_external_knowledge(meme_text: str):
    """
    Return neutral lexicon cues for the teacher prompt.

    This is optional and controlled by TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE.
    It only uses OCR text, not the image pixels, so it is fast and independent
    from the BLIP preprocessing pipeline.
    """
    if not TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE:
        return "disabled", "", 0

    if not EXTERNAL_LEXICON_AVAILABLE:
        warning = f"External lexicon module unavailable: {_EXTERNAL_LEXICON_IMPORT_ERROR}"
        return "requested but unavailable", warning, 0

    result = retrieve_lexicon_prompt_context(
        meme_text=meme_text,
        image_caption="",
        lexicon_dir=EXTERNAL_LEXICON_DIR,
        max_items=TEACHER_EXTERNAL_KNOWLEDGE_MAX_ITEMS,
        min_score=TEACHER_EXTERNAL_KNOWLEDGE_MIN_SCORE,
        include_builtin=True,
    )

    prompt_text = result["prompt_text"]
    count = int(result["num_candidates"])

    if count == 0:
        return "enabled; no relevant cues retrieved", prompt_text, count

    return f"enabled; {count} candidate cue(s) retrieved", prompt_text, count


def build_messages(meme_text: str):
    external_status, external_text, external_count = get_teacher_external_knowledge(meme_text)

    if TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE:
        external_block = (
            "Retrieved external cues:\n"
            f"{external_text}\n\n"
            "How to use these cues:\n"
            "- Treat them as warning signals only, not labels.\n"
            "- Use them to check for possible protected-group targeting, implicit stereotypes, coded meaning, or dehumanizing associations.\n"
            "- Do not classify as hateful from a cue alone.\n"
            "- If a cue only indicates that an identity term appears, require an additional negative claim, stereotype, contempt, threat, exclusion, or dehumanization before choosing hateful.\n"
            "- If the cue is irrelevant to the image/text meaning, ignore it.\n\n"
        )
    else:
        external_block = "Not provided.\n\n"

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
        "external_knowledge_text": external_text,
        "external_knowledge_count": external_count,
    }


# ============================================================
# 3. Parsing and validation
# ============================================================

# These caps prevent the teacher labels from becoming hard 0/1 targets.
# The model's written confidence is not perfectly calibrated, so keeping a
# small amount of probability mass on the other class is safer for distillation.
CONFIDENCE_DEFAULT = 0.75
CONFIDENCE_MIN = 0.50
CONFIDENCE_MAX = 0.97


def extract_field(text: str, field_name: str) -> str:
    """
    Extract one field from the teacher's structured text output.

    Example:
        Confidence: 0.82
        Target: immigrants
    """
    field_names = [
        "Label",
        "Confidence",
        "Target",
        "Protected characteristic",
        "Hateful cue",
        "Explanation",
    ]

    next_fields = "|".join(re.escape(name) for name in field_names if name != field_name)

    pattern = (
        rf"{re.escape(field_name)}\s*:\s*"
        rf"(.*?)"
        rf"(?=\n\s*(?:{next_fields})\s*:|\Z)"
    )

    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_confidence(text: str) -> float:
    """
    Parse the teacher's self-reported confidence and convert it to a safe range.

    Accepted examples:
        Confidence: 0.82
        Confidence: 82%
        Confidence: 82
    """
    confidence_text = extract_field(text, "Confidence")

    if not confidence_text:
        return CONFIDENCE_DEFAULT

    number_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", confidence_text)

    if not number_match:
        return CONFIDENCE_DEFAULT

    confidence = float(number_match.group(1))

    # Accept either 0.82 or 82 / 82%.
    if confidence > 1.0:
        confidence = confidence / 100.0

    # Clamp to keep labels soft.
    confidence = max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, confidence))

    return float(confidence)


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

    # Convert label + confidence into a soft two-class distribution.
    # If Label=hateful and Confidence=0.82:
    #     prob_hate=0.82, prob_not_hate=0.18
    # If Label=not-hateful and Confidence=0.82:
    #     prob_not_hate=0.82, prob_hate=0.18
    if is_hate:
        prob_hate = confidence
        prob_not_hate = 1.0 - confidence
        predicted_label = 1
    else:
        prob_not_hate = confidence
        prob_hate = 1.0 - confidence
        predicted_label = 0

    # Round only for cleaner JSONL files.
    prob_hate = round(float(prob_hate), 6)
    prob_not_hate = round(float(prob_not_hate), 6)
    confidence = round(float(confidence), 6)

    # Safety check.
    total = prob_hate + prob_not_hate
    if abs(total - 1.0) > 1e-5:
        prob_hate = prob_hate / total
        prob_not_hate = prob_not_hate / total

    rationale = extract_field(text, "Explanation")
    target = extract_field(text, "Target") or "unclear"
    protected_characteristic = extract_field(text, "Protected characteristic") or "unclear"
    hateful_cue = extract_field(text, "Hateful cue") or "none"

    # ------------------------------------------------------------
    # Consistency guard:
    # Do not trust outputs where the label says hateful but the
    # structured fields say there is no target/protected group/cue.
    # This catches cases like:
    # Label: hateful
    # Target: none
    # Protected characteristic: none
    # Hateful cue: none
    # ------------------------------------------------------------
    target_norm = target.strip().lower()
    protected_norm = protected_characteristic.strip().lower()
    cue_norm = hateful_cue.strip().lower()
    rationale_norm = rationale.strip().lower()

    weak_target = target_norm in {"", "none", "unclear", "n/a", "na"}
    weak_protected = protected_norm in {"", "none", "unclear", "n/a", "na"}
    weak_cue = cue_norm in {"", "none", "unclear", "n/a", "na"}

    explanation_denies_hate = (
        "no protected target" in rationale_norm
        or "no explicit or implicit targeting" in rationale_norm
        or "no hateful cue" in rationale_norm
        or "lacks both a targeted group and hateful cues" in rationale_norm
        or "does not meet the rubric for hateful content" in rationale_norm
    )

    inconsistent_hateful = (
        predicted_label == 1
        and (
            (weak_target and weak_protected)
            or (weak_protected and weak_cue)
            or (weak_target and weak_cue)
            or explanation_denies_hate
        )
    )

    if inconsistent_hateful:
        return {
            "prob_not_hate": 0.5,
            "prob_hate": 0.5,
            "predicted_label": -1,
            "confidence": 0.5,
            "target": target,
            "protected_characteristic": protected_characteristic,
            "hateful_cue": hateful_cue,
            "rationale": rationale,
            "parse_failed": True,
            "parse_error": "INCONSISTENT_HATEFUL_OUTPUT",
        }

    return {
        "prob_not_hate": prob_not_hate,
        "prob_hate": prob_hate,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "target": target,
        "protected_characteristic": protected_characteristic,
        "hateful_cue": hateful_cue,
        "rationale": rationale,
        "parse_failed": False,
        "parse_error": None,
    }


def fallback_teacher_result(raw_output: str, error_message: str):
    """
    Fallback used if JSON parsing fails.

    It creates a neutral soft label so the script does not crash. The raw output
    is still saved for later inspection.
    """
    return {
        "prob_not_hate": 0.5,
        "prob_hate": 0.5,
        "predicted_label": -1,
        "confidence": 0.5,
        "rationale": "PARSE_FAILED",
        "parse_failed": True,
        "parse_error": error_message,
        "raw_output": raw_output,
    }


# ============================================================
# 4. Model loading and inference
# ============================================================

def load_teacher_model_and_processor():
    print("\n================ Loading MemeLens teacher ================")
    print("Model:", TEACHER_MODEL_NAME)
    print("HF cache:", HF_CACHE_DIR)

    processor = AutoProcessor.from_pretrained(
        TEACHER_MODEL_NAME,
        cache_dir=str(HF_CACHE_DIR),
        local_files_only=True,
    )
    # Cap image resolution: Qwen2VLImageProcessor falls back to size["longest_edge"]
    # when no max_pixels kwarg is passed. ~401k px avoids "too many resources" on Volta.
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
    """
    Pick a safe device for inputs when model is loaded with device_map='auto'.
    """
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
# 5. Main
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
    print("EXTERNAL_LEXICON_DIR:", EXTERNAL_LEXICON_DIR)
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
        row_id = row["id"]

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
                "source": row["source"],
                "teacher_external_knowledge_included": bool(TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE),
                "teacher_external_knowledge_status": external_prompt_metadata.get("external_knowledge_status"),
                "teacher_external_knowledge_count": int(external_prompt_metadata.get("external_knowledge_count", 0)),
                "teacher_external_knowledge_text": external_prompt_metadata.get("external_knowledge_text", ""),
                **normalized,
            }

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
                "source": row.get("source", "unknown"),
                "teacher_external_knowledge_included": bool(TEACHER_INCLUDE_EXTERNAL_KNOWLEDGE),
                "teacher_external_knowledge_status": "error before/during teacher call",
                "teacher_external_knowledge_count": 0,
                "teacher_external_knowledge_text": "",
                "prob_not_hate": 0.5,
                "prob_hate": 0.5,
                "predicted_label": -1,
                "confidence": 0.5,
                "target": "unclear",
                "protected_characteristic": "unclear",
                "hateful_cue": "none",
                "rationale": "TEACHER_INFERENCE_FAILED",
                "parse_failed": True,
                "parse_error": str(error),
                "raw_output": "",
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
