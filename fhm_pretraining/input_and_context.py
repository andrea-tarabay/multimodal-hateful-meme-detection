"""
Meme preprocessing pipeline.

This script combines two parts of the preprocessing code:
1. Image processing + BLIP image caption generation + text tokenization.
2. External knowledge retrieval from GenericsKB and lexicons. 

NOTE: WE ONLY USE DATA DROM GENERICSKB FOR THE FINAL RESULTS, BECAUSE WIKIDATA AND WIKIPEDIA RETRIEVAL HAD LOW QUALITY AND DID NOT IMPROVE THE MODEL. 


The main function to call is:
    process_dataset_row(row)

Given a dataset row like:
    {
        "id": "fhm_42953",
        "image_path": "images/fhm_42953.png",
        "text": "its their character not their color that matters",
        "label": 0,
        "source": "fhm"
    }

It returns tensors compatible with train.py:
    pixel_values           shape [3, H, W]
    input_ids              shape [MAX_TEXT_LENGTH]
    attention_mask         shape [MAX_TEXT_LENGTH]
    clip_text_embedding    shape [CLIP_TEXT_DIM]
    knowledge_embeddings   shape [KNOWLEDGE_TOP_K, 384]
    knowledge_mask         shape [KNOWLEDGE_TOP_K]
    labels                 scalar tensor

After PyTorch DataLoader batching, these become:
    pixel_values           shape [B, 3, H, W]
    input_ids              shape [B, MAX_TEXT_LENGTH]
    attention_mask         shape [B, MAX_TEXT_LENGTH]
    clip_text_embedding    shape [B, CLIP_TEXT_DIM]
    knowledge_embeddings   shape [B, KNOWLEDGE_TOP_K, 384]
    knowledge_mask         shape [B, KNOWLEDGE_TOP_K]
    labels                 shape [B]
"""

# ============================================================
# 0. Imports
# ============================================================

import re
import time
import math
import os
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn.functional as F

from PIL import Image
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    Blip2Processor,
    CLIPTokenizer,
    CLIPTextModel,
)

# External lexicon source. If this file is missing, the original
try:
    from external_lexicon_knowledge import (
        load_external_lexicon_knowledge,
        collect_external_lexicon_candidates,
        collect_external_lexicon_candidates_indexed,
        build_lexicon_index,
        build_query_terms_for_lexicons,
    )
    EXTERNAL_LEXICON_AVAILABLE = True
except Exception as _external_lexicon_error:
    load_external_lexicon_knowledge = None
    collect_external_lexicon_candidates = None
    collect_external_lexicon_candidates_indexed = None
    build_lexicon_index = None
    build_query_terms_for_lexicons = None
    EXTERNAL_LEXICON_AVAILABLE = False
    _EXTERNAL_LEXICON_IMPORT_ERROR = _external_lexicon_error


# ============================================================
# 1. User variables / configuration
# ============================================================
# I keep the main variables here so I can quickly change them when running
# the script on a cluster.

# ----------------------------
# Paths
# ----------------------------

# Base directory where the dataset images are stored.
# The image_path from each row is joined with this directory.
DATA_ROOT = Path.cwd()

# Example: if the row has image_path="images/fhm_42953.png",
# the final image path will be DATA_ROOT / "images/fhm_42953.png".


# ----------------------------
# Device
# ----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def env_flag(name: str, default: bool) -> bool:
    """Read a boolean from environment variables for easy cluster ablations."""
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# Optional import from config.py. Defaults keep this file runnable standalone.
try:
    from config import (
        CLIP_MODEL_NAME as CONFIG_CLIP_MODEL_NAME,
        CLIP_TEXT_DIM as CONFIG_CLIP_TEXT_DIM,
        USE_CLIP_TEXT_EMBEDDINGS as CONFIG_USE_CLIP_TEXT_EMBEDDINGS,
    )
except Exception:
    CONFIG_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    CONFIG_CLIP_TEXT_DIM = 512
    CONFIG_USE_CLIP_TEXT_EMBEDDINGS = True


# ----------------------------
# Image captioning model
# ----------------------------
# I keep BLIP here only for generating image captions.

CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
CAPTION_MAX_NEW_TOKENS = 30


# ----------------------------
# BLIP2 processor
# ----------------------------
# I use the same BLIP2 processor for both:
#   1. image pixel preprocessing
#   2. text tokenization
#
# This way, pixel_values and tokenized text are both compatible with BLIP2.

BLIP2_PROCESSOR_NAME = "Salesforce/blip2-opt-2.7b"


# ----------------------------
# Text settings
# ----------------------------

MAX_TEXT_LENGTH = 128


# ----------------------------
# CLIP text embedding settings
# ----------------------------
# The old input_ids/attention_mask are still saved for compatibility, but the
# student models will use this pretrained CLIP text embedding instead of a
# randomly initialized nn.Embedding text branch.

CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", CONFIG_CLIP_MODEL_NAME)
CLIP_TEXT_DIM = int(os.getenv("CLIP_TEXT_DIM", str(CONFIG_CLIP_TEXT_DIM)))
USE_CLIP_TEXT_EMBEDDINGS = env_flag(
    "USE_CLIP_TEXT_EMBEDDINGS",
    CONFIG_USE_CLIP_TEXT_EMBEDDINGS,
)
CLIP_MAX_TEXT_LENGTH = 77


# ----------------------------
# Sentence embedding model
# ----------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ----------------------------
# GenericsKB
# ----------------------------

GENERICS_DATASET_NAME = "community-datasets/generics_kb"
GENERICS_DATASET_CONFIG = "generics_kb_best"
GENERICS_DATASET_SPLIT = "train"


# ----------------------------
# Query term extraction
# ----------------------------

MAX_QUERY_TERMS = 12


# ----------------------------
# Wikidata retrieval settings
# ----------------------------

USE_WIKIDATA = False
WIKIDATA_RESULTS_PER_TERM = 5
USE_WIKIDATA_EXACT_LABEL_FILTER = True
WIKIDATA_SLEEP = 1.0
WIKIDATA_MAX_RETRIES = 4


# ----------------------------
# Wikipedia retrieval settings
# ----------------------------

USE_WIKIPEDIA = False
WIKIPEDIA_RESULTS_PER_TERM = 2
WIKIPEDIA_SENTENCES_PER_PAGE = 1
WIKIPEDIA_EXACT_TITLE_MATCH_ONLY = True
WIKIPEDIA_MAX_CONTEXT_WORDS = 25
WIKIPEDIA_LANGUAGE = "en"
WIKIPEDIA_SLEEP = 1.5
WIKIPEDIA_MAX_RETRIES = 4


# ----------------------------
# GenericsKB retrieval settings
# ----------------------------

USE_GENERICSKB = True
GENERICS_RESULTS_PER_TERM = 20
MIN_GENERICS_SCORE = 0.0



# ----------------------------
# External lexicon retrieval settings
# ----------------------------
# This adds a lightweight local source for social/offensive lexical cues.
# Keep the lexicon code and raw lexicon files separate from this main pipeline.
# Put files in: external_lexicons/
#   xenophobia_lexicon_en.txt, immigrant_lexicon_en.txt,
#   insults_lexicon_en.txt, misogyny_lexicon_en.txt, mol.csv
USE_EXTERNAL_LEXICONS = env_flag("USE_EXTERNAL_LEXICONS", True)
EXTERNAL_LEXICON_DIR = Path(os.getenv(
    "EXTERNAL_LEXICON_DIR",
    str(Path(__file__).resolve().parent / "external_lexicons")
))
EXTERNAL_LEXICON_RESULTS_PER_TERM = int(os.getenv("EXTERNAL_LEXICON_RESULTS_PER_TERM", "20"))
EXTERNAL_LEXICON_MIN_SCORE = float(os.getenv("EXTERNAL_LEXICON_MIN_SCORE", "0.55"))

# ----------------------------
# Final ranking weights
# ----------------------------

ALPHA_BM25 = 0.45
BETA_EMBEDDING = 0.50
GAMMA_SOURCE = 0.05


# ----------------------------
# Knowledge embedding settings
# ----------------------------

KNOWLEDGE_TOP_K = 5
NORMALIZE_KNOWLEDGE_EMBEDDINGS = True


# ----------------------------
# API user agent
# ----------------------------
# This is used when calling Wikidata and Wikipedia.

USER_AGENT = "MemeExternalKnowledgeRetrieval/0.7 academic project"


# ============================================================
# 2. Load models and datasets once
# ============================================================
# These objects are loaded globally so that they are not reloaded for every
# dataset row. This is important when running preprocessing for many samples.

# ------------------------------------------------------------
# 2.1 Load BLIP captioning model
# ------------------------------------------------------------

caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_NAME)
caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME, use_safetensors=True).to(DEVICE)
caption_model.eval()

# ------------------------------------------------------------
# 2.2 Load BLIP2 processor
# ------------------------------------------------------------
# I use this same processor for both image preprocessing and text tokenization.

blip2_processor = Blip2Processor.from_pretrained(BLIP2_PROCESSOR_NAME)
text_tokenizer = blip2_processor.tokenizer

# Some tokenizers do not have a pad token set by default.
# If that happens, I set the pad token to the EOS token so padding works.
if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token

# ------------------------------------------------------------
# 2.3 Load CLIP text encoder for pretrained meme-text embeddings
# ------------------------------------------------------------

if USE_CLIP_TEXT_EMBEDDINGS:
    clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
    clip_text_model = CLIPTextModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True).to(DEVICE)
    clip_text_model.eval()
else:
    clip_tokenizer = None
    clip_text_model = None

# ------------------------------------------------------------
# 2.4 Load spaCy and sentence-transformer
# ------------------------------------------------------------

nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ------------------------------------------------------------
# 2.5 Load GenericsKB once
# ------------------------------------------------------------

generics_ds = load_dataset(
    GENERICS_DATASET_NAME,
    GENERICS_DATASET_CONFIG,
    split=GENERICS_DATASET_SPLIT,
)

generics_df = generics_ds.to_pandas()


# ============================================================
# 3. General helpers
# ============================================================

def count_parameters(model):
    """
    Count the total and trainable parameters of a model.
    This is kept here in case I want to check model size later.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def normalize_genericskb_dataframe(df):
    """
    Normalize GenericsKB columns into the column names used by this script.

    The final columns needed are:
        term
        generic_sentence
        score
        term_norm
    """

    df = df.copy()

    if "generic_sentence" not in df.columns and "sentence" in df.columns:
        df["generic_sentence"] = df["sentence"]

    if "term" not in df.columns and "concept_name" in df.columns:
        df["term"] = df["concept_name"]

    if "score" not in df.columns and "bert_score" in df.columns:
        df["score"] = df["bert_score"]

    if "score" not in df.columns:
        df["score"] = 1.0

    required = ["term", "generic_sentence", "score"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required GenericsKB columns: {missing}")

    df["term"] = df["term"].astype(str)
    df["generic_sentence"] = df["generic_sentence"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)

    df["term_norm"] = df["term"].str.lower().str.strip()

    return df


# Normalize GenericsKB after defining the helper.
generics_df = normalize_genericskb_dataframe(generics_df)

# Pre-index by term_norm for O(1) lookup, replacing O(750K) full scans per query term.
_generics_index: dict = {
    term_norm: group.sort_values("score", ascending=False).reset_index(drop=True)
    for term_norm, group in generics_df.groupby("term_norm")
}

# Load external lexical/social cue knowledge once. If unavailable, use an empty table.
if USE_EXTERNAL_LEXICONS and not EXTERNAL_LEXICON_AVAILABLE:
    print(f"[Warning] USE_EXTERNAL_LEXICONS=True but external_lexicon_knowledge.py could not be imported: {_EXTERNAL_LEXICON_IMPORT_ERROR}")

if EXTERNAL_LEXICON_AVAILABLE and USE_EXTERNAL_LEXICONS:
    external_lexicon_df = load_external_lexicon_knowledge(EXTERNAL_LEXICON_DIR)
    print(f"[Info] External lexicon source enabled: {len(external_lexicon_df)} rows from {EXTERNAL_LEXICON_DIR}")
    _external_lexicon_index: dict = build_lexicon_index(external_lexicon_df)
else:
    external_lexicon_df = pd.DataFrame(columns=[
        "term", "aliases", "category", "sentence", "score",
        "source_name", "source_file", "citation_key", "term_norm", "aliases_list"
    ])
    _external_lexicon_index: dict = {}



# ============================================================
# 4. Image and text preprocessing
# ============================================================

def load_image(image_path):
    """
    Load an image from disk and convert it to RGB.
    This makes sure the image has 3 channels before it goes into the model.
    """
    image = Image.open(image_path).convert("RGB")
    return image


def generate_image_caption(image, max_new_tokens=CAPTION_MAX_NEW_TOKENS):
    """
    Generate a short caption for an image using BLIP.
    The caption is later used together with the meme text for knowledge retrieval.
    """
    inputs = caption_processor(
        images=image,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = caption_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    caption = caption_processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    return caption


def image_to_pixel_values(image):
    """
    Convert a PIL image into BLIP2 pixel_values.

    The output shape is usually:
        [1, 3, H, W]

    Example:
        [1, 3, 224, 224]
    or depending on the processor configuration.
    """
    encoded_image = blip2_processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = encoded_image["pixel_values"]

    return pixel_values


def tokenize_text(text, max_length=MAX_TEXT_LENGTH):
    """
    Convert raw meme text into input_ids and attention_mask using the
    same BLIP2 tokenizer that belongs to the BLIP2 processor.

    input_ids:
        Token IDs used by the model.

    attention_mask:
        1 for real tokens and 0 for padding tokens.
    """
    encoded_text = text_tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = encoded_text["input_ids"]
    attention_mask = encoded_text["attention_mask"]

    return input_ids, attention_mask


def text_to_clip_embedding(text, max_length=CLIP_MAX_TEXT_LENGTH):
    """
    Convert raw meme text into a pretrained CLIP text embedding.

    This is the representation used by the updated student models.
    The output shape is [1, CLIP_TEXT_DIM], usually [1, 512] for
    openai/clip-vit-base-patch32.
    """
    if not USE_CLIP_TEXT_EMBEDDINGS:
        return torch.zeros(1, CLIP_TEXT_DIM, dtype=torch.float32)

    if clip_tokenizer is None or clip_text_model is None:
        raise RuntimeError("CLIP text encoder is not loaded.")

    encoded_text = clip_tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = clip_text_model(**encoded_text)
        clip_text_embedding = outputs.pooler_output.float()
        clip_text_embedding = F.normalize(clip_text_embedding, p=2, dim=-1)

    if clip_text_embedding.shape[-1] != CLIP_TEXT_DIM:
        raise ValueError(
            f"Expected CLIP_TEXT_DIM={CLIP_TEXT_DIM}, "
            f"but got {clip_text_embedding.shape[-1]}."
        )

    return clip_text_embedding.cpu()


def process_meme_sample(image_path, meme_text=None, max_text_length=MAX_TEXT_LENGTH):
    """
    Do the basic preprocessing for one meme.

    This function:
    1. Loads the image.
    2. Converts the image into pixel_values using the BLIP2 processor.
    3. Generates an image caption with BLIP.
    4. Uses the meme text from the dataset as OCR text.
    5. Tokenizes the meme text using the BLIP2 tokenizer.
    6. Creates a pretrained CLIP text embedding for the meme text.

    If meme_text is None, this function raises an error because this script
    assumes that the dataset already provides the text.
    """

    # 1. Load image
    image = load_image(image_path)

    # 2. Convert image to pixel values
    pixel_values = image_to_pixel_values(image)

    # 3. Generate image caption
    image_caption = generate_image_caption(image)

    # 4. Load OCR/meme text
    if meme_text is None:
        raise ValueError(
            "No meme_text provided. If your dataset does not include meme text, "
            "you need OCR. If it does include meme text, pass it as meme_text."
        )

    ocr_text = meme_text

    # 5. Tokenize OCR/meme text with the BLIP2 tokenizer.
    # These old fields are kept for compatibility/debugging.
    input_ids, attention_mask = tokenize_text(
        ocr_text,
        max_length=max_text_length
    )

    # 6. Create pretrained CLIP text embedding used by the updated models.
    clip_text_embedding = text_to_clip_embedding(ocr_text)

    return {
        "image": image,
        "pixel_values": pixel_values,
        "image_caption": image_caption,
        "ocr_text": ocr_text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "clip_text_embedding": clip_text_embedding,
    }


# ============================================================
# 5. General text utilities
# ============================================================

GENERIC_BAD_TERMS = {
    "image", "meme", "caption", "picture", "photo", "scene",
    "thing", "someone", "something", "anything", "meme image",
    "background", "foreground", "text", "word", "words",

    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",

    "this", "that", "these", "those", "there", "here",

    "person", "people", "man", "woman", "men", "women",
    "guy", "girl", "boy",

    "show", "showing", "shown",
    "stand", "standing",
    "sit", "sitting",
    "look", "looking",
    "appear", "appearing",
    "depict", "depicting",
    "display", "displaying",

    "pile", "piles",
}


def normalize_text(text: str) -> str:
    """
    Lowercase text and remove extra whitespace.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def simple_tokenize(text: str):
    """
    Simple tokenizer used for BM25.
    It keeps alphabetic words and lowercases them.
    """
    return re.findall(r"\b[a-zA-Z][a-zA-Z\-]+\b", text.lower())


def unique_preserve_order(items):
    """
    Remove duplicates while keeping the original order.
    """
    output = []
    seen = set()

    for item in items:
        if item is None:
            continue

        key = item.lower().strip()

        if key and key not in seen:
            output.append(item)
            seen.add(key)

    return output


def strip_edge_connectors(term: str):
    """
    Remove connector words from the beginning or end of a term.
    For example, this avoids keeping terms like "of money" or "money and".
    """
    if not term:
        return term

    words = term.strip().split()

    while words and words[0].lower() in {"of", "and", "or", "&"}:
        words = words[1:]

    while words and words[-1].lower() in {"of", "and", "or", "the", "&"}:
        words = words[:-1]

    return " ".join(words)


def clean_term(term: str):
    """
    Clean one candidate query term.
    Very generic or very short terms are removed.
    """
    if not term:
        return None

    term = term.strip()
    term = re.sub(r"\s+", " ", term)
    term = strip_edge_connectors(term)

    if len(term) < 3:
        return None

    if term.lower() in GENERIC_BAD_TERMS:
        return None

    return term


def safe_lemma(token):
    """
    Lemmatize a spaCy token, but keep some words unchanged.
    Some words like "media" can be badly changed by lemmatization.
    """
    lower = token.text.lower()

    preserve_as_is = {
        "media",
        "news",
        "police",
    }

    if lower in preserve_as_is:
        return lower

    return token.lemma_.lower()


# ============================================================
# 6. Query term extraction
# ============================================================

def is_connector_token(token):
    """
    Check if a token is allowed inside a proper-noun phrase.
    """
    return token.text.lower() in {"of", "the", "and", "&"}


def extract_title_or_propn_phrases(doc):
    """
    Extract proper-noun phrases from a spaCy document.
    This helps keep names like "Star of David" together.
    """
    phrases = []
    current = []

    def flush():
        nonlocal current

        if current:
            phrase = " ".join(current).strip()
            phrase = re.sub(r"\s+", " ", phrase)
            phrase = strip_edge_connectors(phrase)
            phrase = clean_term(phrase)

            if phrase:
                phrases.append(phrase)

        current = []

    for token in doc:
        if token.is_punct:
            flush()
            continue

        if token.pos_ == "PROPN":
            current.append(token.text)

        elif current and is_connector_token(token):
            current.append(token.text)

        else:
            flush()

    flush()

    return phrases


def remove_subterms(terms):
    """
    Remove terms that are already included inside longer terms.
    For example, if "Star of David" exists, then "David" can be removed.
    """
    terms = unique_preserve_order(terms)
    final_terms = []

    for term in terms:
        term_lower = term.lower()
        is_subterm = False

        for other in terms:
            other_lower = other.lower()

            if term_lower == other_lower:
                continue

            pattern = r"\b" + re.escape(term_lower) + r"\b"

            if re.search(pattern, other_lower) and len(other_lower) > len(term_lower):
                is_subterm = True
                break

        if not is_subterm:
            final_terms.append(term)

    return final_terms


def extract_query_terms(
    meme_text: str,
    image_caption: str,
    max_terms: int = MAX_QUERY_TERMS,
    include_pos={"NOUN", "PROPN", "VERB", "ADJ"}
):
    """
    Extract query terms from the meme text and the generated image caption.

    The function uses:
    1. Named entities.
    2. Proper noun phrases.
    3. Noun chunks.
    4. Individual useful tokens.
    """
    text = f"{meme_text}. {image_caption}"
    doc = nlp(text)

    terms = []

    for ent in doc.ents:
        cleaned = clean_term(ent.text)
        if cleaned:
            terms.append(cleaned)

    for phrase in extract_title_or_propn_phrases(doc):
        cleaned = clean_term(phrase)
        if cleaned:
            terms.append(cleaned)

    for chunk in doc.noun_chunks:
        cleaned_tokens = []

        for token in chunk:
            if token.is_punct:
                continue

            if token.pos_ == "DET":
                continue

            if token.is_stop and token.text.lower() not in {"of", "and"}:
                continue

            if token.pos_ == "PROPN":
                cleaned_tokens.append(token.text)

            elif token.pos_ in {"NOUN", "ADJ"}:
                cleaned_tokens.append(safe_lemma(token))

            elif token.text.lower() in {"of", "and"}:
                cleaned_tokens.append(token.text.lower())

        phrase = " ".join(cleaned_tokens)
        phrase = clean_term(phrase)

        if phrase:
            terms.append(phrase)

    for token in doc:
        if token.is_stop:
            continue

        if token.is_punct:
            continue

        if token.pos_ not in include_pos:
            continue

        if len(token.text) < 3:
            continue

        if token.pos_ == "PROPN":
            term = token.text
        else:
            term = safe_lemma(token)

        term = clean_term(term)

        if term:
            terms.append(term)

    terms = unique_preserve_order(terms)
    terms = remove_subterms(terms)

    return terms[:max_terms]


# ============================================================
# 7. Wikidata retrieval
# ============================================================

WIKIDATA_CACHE = {}


def wikidata_entity_to_sentence(label: str, description: str):
    """
    Convert a Wikidata label and description into one sentence.
    """
    label = str(label).strip()
    description = str(description).strip()
    description = description.rstrip(".")
    description = re.sub(r"\s+", " ", description)

    if not label or not description:
        return None

    desc_lower = description.lower()

    if desc_lower.startswith(("a ", "an ", "the ")):
        sentence = f"{label} is {description}."
    else:
        sentence = f"{label} is a {description}."

    sentence = re.sub(r"\s+", " ", sentence).strip()

    return sentence


def search_wikidata(
    term: str,
    limit: int = WIKIDATA_RESULTS_PER_TERM,
    sleep: float = WIKIDATA_SLEEP,
    max_retries: int = WIKIDATA_MAX_RETRIES
):
    """
    Search Wikidata for one query term.
    Results are cached so repeated terms do not call the API again.
    """
    cache_key = (term.lower().strip(), limit)

    if cache_key in WIKIDATA_CACHE:
        return WIKIDATA_CACHE[cache_key]

    url = "https://www.wikidata.org/w/api.php"

    params = {
        "action": "wbsearchentities",
        "search": term,
        "language": "en",
        "format": "json",
        "limit": limit,
        "type": "item",
    }

    headers = {
        "User-Agent": USER_AGENT
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_time = float(retry_after) if retry_after is not None else sleep * (2 ** attempt) + 2.0

                print(
                    f"[Warning] 429 Too Many Requests for Wikidata term={term!r}. "
                    f"Waiting {wait_time:.1f}s and retrying..."
                )

                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()
            results = data.get("search", [])

            WIKIDATA_CACHE[cache_key] = results

            time.sleep(sleep)
            return results

        except requests.RequestException as error:
            wait_time = sleep * (2 ** attempt) + 2.0

            print(
                f"[Warning] Wikidata search failed for term={term!r}: {error}. "
                f"Waiting {wait_time:.1f}s and retrying..."
            )

            time.sleep(wait_time)

    print(f"[Warning] Giving up on Wikidata term={term!r}")
    WIKIDATA_CACHE[cache_key] = []

    return []


def collect_wikidata_candidates(
    query_terms,
    results_per_term: int = WIKIDATA_RESULTS_PER_TERM,
    use_exact_label_filter: bool = USE_WIKIDATA_EXACT_LABEL_FILTER
):
    """
    Collect Wikidata candidate context sentences for all query terms.
    """
    candidates = []

    for term in query_terms:
        results = search_wikidata(term, limit=results_per_term)

        for result in results:
            qid = result.get("id")
            label = result.get("label")
            description = result.get("description")
            wikidata_score = result.get("score", 0.0)

            if not qid or not label or not description:
                continue

            if use_exact_label_filter and label.strip() != term.strip():
                continue

            sentence = wikidata_entity_to_sentence(label, description)

            if not sentence:
                continue

            candidates.append({
                "source": "wikidata",
                "matched_query": term,
                "label_or_term": label,
                "sentence": sentence,
                "source_score": float(wikidata_score),
                "url": f"https://www.wikidata.org/wiki/{qid}",
                "qid": qid,
                "raw_description": description,
            })

    return candidates


# ============================================================
# 8. Wikipedia retrieval
# ============================================================

WIKIPEDIA_SEARCH_CACHE = {}
WIKIPEDIA_SUMMARY_CACHE = {}


def search_wikipedia_pages(
    term: str,
    limit: int = WIKIPEDIA_RESULTS_PER_TERM,
    language: str = WIKIPEDIA_LANGUAGE,
    sleep: float = WIKIPEDIA_SLEEP,
    max_retries: int = WIKIPEDIA_MAX_RETRIES
):
    """
    Search Wikipedia pages for one query term.
    Results are cached to reduce API calls.
    """
    cache_key = (language, term.lower().strip(), limit)

    if cache_key in WIKIPEDIA_SEARCH_CACHE:
        return WIKIPEDIA_SEARCH_CACHE[cache_key]

    url = f"https://{language}.wikipedia.org/w/rest.php/v1/search/page"

    params = {
        "q": term,
        "limit": limit,
    }

    headers = {
        "User-Agent": USER_AGENT
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_time = float(retry_after) if retry_after else sleep * (2 ** attempt) + 2.0

                print(
                    f"[Warning] 429 Too Many Requests for Wikipedia search term={term!r}. "
                    f"Waiting {wait_time:.1f}s and retrying..."
                )

                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            pages = data.get("pages", [])
            WIKIPEDIA_SEARCH_CACHE[cache_key] = pages

            time.sleep(sleep)
            return pages

        except requests.RequestException as error:
            wait_time = sleep * (2 ** attempt) + 2.0

            print(
                f"[Warning] Wikipedia search failed for term={term!r}: {error}. "
                f"Waiting {wait_time:.1f}s and retrying..."
            )

            time.sleep(wait_time)

    print(f"[Warning] Giving up on Wikipedia search term={term!r}")
    WIKIPEDIA_SEARCH_CACHE[cache_key] = []

    return []


def get_wikipedia_summary(
    title: str,
    language: str = WIKIPEDIA_LANGUAGE,
    sleep: float = WIKIPEDIA_SLEEP,
    max_retries: int = WIKIPEDIA_MAX_RETRIES
):
    """
    Get a short Wikipedia summary for a page title.
    The result is cached so each title is only requested once.
    """
    cache_key = (language, title)

    if cache_key in WIKIPEDIA_SUMMARY_CACHE:
        return WIKIPEDIA_SUMMARY_CACHE[cache_key]

    safe_title = title.replace(" ", "_")
    url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{safe_title}"

    headers = {
        "User-Agent": USER_AGENT
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code == 404:
                WIKIPEDIA_SUMMARY_CACHE[cache_key] = None
                return None

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_time = float(retry_after) if retry_after else sleep * (2 ** attempt) + 2.0

                print(
                    f"[Warning] 429 Too Many Requests for Wikipedia summary title={title!r}. "
                    f"Waiting {wait_time:.1f}s and retrying..."
                )

                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            WIKIPEDIA_SUMMARY_CACHE[cache_key] = data

            time.sleep(sleep)
            return data

        except requests.RequestException as error:
            wait_time = sleep * (2 ** attempt) + 2.0

            print(
                f"[Warning] Wikipedia summary failed for title={title!r}: {error}. "
                f"Waiting {wait_time:.1f}s and retrying..."
            )

            time.sleep(wait_time)

    print(f"[Warning] Giving up on Wikipedia summary title={title!r}")
    WIKIPEDIA_SUMMARY_CACHE[cache_key] = None

    return None


def is_bad_wikipedia_sentence(sentence: str) -> bool:
    """
    Remove Wikipedia sentences that are not useful as context.
    These are usually disambiguation, list, or very short sentences.
    """
    if not sentence:
        return True

    s = sentence.strip().lower()

    bad_patterns = [
        "may refer to:",
        "may refer to",
        "can refer to:",
        "can refer to",
        "refers to:",
        "refers to",
        "is a list of",
        "list of",
        "index of",
        "set index",
        "outline of",
        "glossary of",
    ]

    if any(pattern in s for pattern in bad_patterns):
        return True

    if len(sentence.split()) < 5:
        return True

    return False


def split_summary_into_sentences(
    text: str,
    max_sentences: int = WIKIPEDIA_SENTENCES_PER_PAGE,
    max_context_words: int = WIKIPEDIA_MAX_CONTEXT_WORDS
):
    """
    Split a Wikipedia summary into short context sentences.
    Only the first useful sentence is normally kept.
    """
    if not text:
        return []

    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)

    cleaned = []

    for sent in sentences:
        sent = sent.strip()

        if is_bad_wikipedia_sentence(sent):
            continue

        words = sent.split()

        if len(words) > max_context_words:
            sent = " ".join(words[:max_context_words]).rstrip(",;:") + "."

        if is_bad_wikipedia_sentence(sent):
            continue

        cleaned.append(sent)

        if len(cleaned) >= max_sentences:
            break

    return cleaned


def is_exact_or_casefold_title_match(query_term: str, page_title: str):
    """
    Check whether the Wikipedia page title exactly matches the query term,
    ignoring capitalization.
    """
    return query_term.strip().lower() == page_title.strip().lower()


def collect_wikipedia_candidates(
    query_terms,
    search_results_per_term: int = WIKIPEDIA_RESULTS_PER_TERM,
    summary_sentences_per_page: int = WIKIPEDIA_SENTENCES_PER_PAGE,
    max_context_words: int = WIKIPEDIA_MAX_CONTEXT_WORDS,
    exact_title_match_only: bool = WIKIPEDIA_EXACT_TITLE_MATCH_ONLY,
    language: str = WIKIPEDIA_LANGUAGE
):
    """
    Collect Wikipedia candidate context sentences for all query terms.
    """
    candidates = []

    for term in query_terms:
        pages = search_wikipedia_pages(
            term=term,
            limit=search_results_per_term,
            language=language
        )

        for rank, page in enumerate(pages):
            title = page.get("title")
            description = page.get("description")

            if not title:
                continue

            if exact_title_match_only and not is_exact_or_casefold_title_match(term, title):
                continue

            summary_data = get_wikipedia_summary(title=title, language=language)

            if not summary_data:
                continue

            extract = summary_data.get("extract")
            page_url = summary_data.get("content_urls", {}).get("desktop", {}).get("page")

            if not extract:
                continue

            summary_sentences = split_summary_into_sentences(
                extract,
                max_sentences=summary_sentences_per_page,
                max_context_words=max_context_words
            )

            for sent_idx, sent in enumerate(summary_sentences):
                if is_bad_wikipedia_sentence(sent):
                    continue

                candidates.append({
                    "source": "wikipedia",
                    "matched_query": term,
                    "label_or_term": title,
                    "sentence": sent,
                    "source_score": 1.0 / (1.0 + rank + sent_idx),
                    "url": page_url,
                    "qid": None,
                    "raw_description": description,
                })

    return candidates


# ============================================================
# 9. GenericsKB retrieval
# ============================================================

def search_genericskb(
    term: str,
    generics_df: pd.DataFrame,
    limit: int = GENERICS_RESULTS_PER_TERM,
    min_score: float = MIN_GENERICS_SCORE
):
    """
    Search GenericsKB for exact matches to one query term.
    Uses the pre-built _generics_index for O(1) lookup instead of a full table scan.
    """
    term_norm = term.lower().strip()

    group = _generics_index.get(term_norm)
    if group is None or group.empty:
        return []

    if min_score > 0.0:
        group = group[group["score"] >= min_score]

    subset = group.head(limit)

    candidates = []

    for _, row in subset.iterrows():
        sentence = str(row["generic_sentence"]).strip()

        if not sentence:
            continue

        candidates.append({
            "source": "genericskb",
            "matched_query": term,
            "label_or_term": row["term"],
            "sentence": sentence,
            "source_score": float(row["score"]),
            "url": None,
            "qid": None,
            "raw_description": None,
        })

    return candidates


def collect_genericskb_candidates(
    query_terms,
    generics_df: pd.DataFrame,
    results_per_term: int = GENERICS_RESULTS_PER_TERM,
    min_score: float = MIN_GENERICS_SCORE
):
    """
    Collect GenericsKB candidate context sentences for all query terms.
    """
    candidates = []

    for term in query_terms:
        results = search_genericskb(
            term=term,
            generics_df=generics_df,
            limit=results_per_term,
            min_score=min_score
        )

        candidates.extend(results)

    return candidates


# ============================================================
# 10. Scoring
# ============================================================

def add_bm25_scores(query: str, candidates):
    """
    Add a BM25 score to each candidate context sentence.
    """
    if not candidates:
        return []

    documents = [item["sentence"] for item in candidates]

    tokenized_docs = [
        simple_tokenize(document)
        for document in documents
    ]

    tokenized_query = simple_tokenize(query)

    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)

    scored = []

    for item, score in zip(candidates, scores):
        new_item = dict(item)
        new_item["bm25_score"] = float(score)
        scored.append(new_item)

    return scored


def add_embedding_scores(query: str, candidates, batch_size: int = 32):
    """
    Add a sentence-transformer cosine similarity score to each candidate.
    """
    if not candidates:
        return []

    candidate_sentences = [item["sentence"] for item in candidates]

    query_embedding = embedding_model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    candidate_embeddings = embedding_model.encode(
        candidate_sentences,
        convert_to_tensor=True,
        normalize_embeddings=True,
        batch_size=batch_size
    )

    scores = util.cos_sim(query_embedding, candidate_embeddings)[0]

    scored = []

    for item, score in zip(candidates, scores):
        new_item = dict(item)
        new_item["embedding_score"] = float(score)
        scored.append(new_item)

    return scored


def minmax_normalize(values):
    """
    Min-max normalize a list of values.
    If all values are the same, return zeros.
    """
    values = np.array(values, dtype=float)

    if len(values) == 0:
        return values

    min_v = values.min()
    max_v = values.max()

    if math.isclose(min_v, max_v):
        return np.zeros_like(values)

    return (values - min_v) / (max_v - min_v)


def add_normalized_source_scores_by_source(candidates):
    """
    Normalize source scores separately for each source.
    This is done because scores from Wikidata, Wikipedia, and GenericsKB
    are not directly comparable.
    """
    if not candidates:
        return []

    df = pd.DataFrame(candidates)

    norm_scores = np.zeros(len(df), dtype=float)

    for source_name in df["source"].unique():
        idx = df.index[df["source"] == source_name].tolist()
        values = df.loc[idx, "source_score"].fillna(0.0).astype(float).values
        norm_values = minmax_normalize(values)

        for pos, row_idx in enumerate(idx):
            norm_scores[row_idx] = norm_values[pos]

    output = []

    for i, item in enumerate(candidates):
        new_item = dict(item)
        new_item["source_score_norm"] = float(norm_scores[i])
        output.append(new_item)

    return output


def combine_scores(
    candidates,
    alpha_bm25: float = ALPHA_BM25,
    beta_embedding: float = BETA_EMBEDDING,
    gamma_source: float = GAMMA_SOURCE
):
    """
    Combine BM25, embedding similarity, and source score into one final score.
    """
    if not candidates:
        return []

    candidates = add_normalized_source_scores_by_source(candidates)

    bm25_values = [item.get("bm25_score", 0.0) for item in candidates]
    emb_values = [item.get("embedding_score", 0.0) for item in candidates]

    bm25_norm = minmax_normalize(bm25_values)
    emb_norm = minmax_normalize(emb_values)

    combined = []

    for i, item in enumerate(candidates):
        new_item = dict(item)

        new_item["bm25_score_norm"] = float(bm25_norm[i])
        new_item["embedding_score_norm"] = float(emb_norm[i])

        new_item["final_score"] = (
            alpha_bm25 * new_item["bm25_score_norm"]
            + beta_embedding * new_item["embedding_score_norm"]
            + gamma_source * new_item["source_score_norm"]
        )

        combined.append(new_item)

    combined.sort(key=lambda x: x["final_score"], reverse=True)

    return combined


# ============================================================
# 11. Select 3 contexts per query term
# ============================================================

def empty_context_for_query_source(query_term: str, source: str):
    """
    Create an empty context row when a source has no result for a query term.
    The context_mask is 0 so the model can ignore it.
    """
    return {
        "source": source,
        "matched_query": query_term,
        "label_or_term": None,
        "sentence": "",
        "source_score": None,
        "source_score_norm": None,
        "bm25_score": None,
        "bm25_score_norm": None,
        "embedding_score": None,
        "embedding_score_norm": None,
        "final_score": None,
        "url": None,
        "qid": None,
        "raw_description": None,
        "context_mask": 0,
    }


def select_top_context_per_source_per_query(
    ranked_candidates,
    query_terms,
    sources=("wikidata", "wikipedia", "genericskb")
):
    """
    For each query term, select one best context from each source.
    This gives up to 3 contexts per query term.
    """
    rows = []

    for query_term in query_terms:
        for source in sources:
            matching = [
                item
                for item in ranked_candidates
                if item.get("matched_query") == query_term
                and item.get("source") == source
            ]

            if matching:
                best = matching[0].copy()
                best["context_mask"] = 1
                rows.append(best)
            else:
                rows.append(empty_context_for_query_source(query_term, source))

    return rows


# ============================================================
# 12. Knowledge embedding creation
# ============================================================

def select_overall_top_k_from_output_df(output_df, top_k: int = KNOWLEDGE_TOP_K):
    """
    Select the top-k real contexts using final_score.
    Empty padded contexts are ignored here.
    """
    real_df = output_df[
        (output_df["context_mask"] == 1)
        & (output_df["final_score"].notna())
        & (output_df["sentence"].astype(str).str.len() > 0)
    ].copy()

    real_df = real_df.sort_values("final_score", ascending=False)

    return real_df.head(top_k)


def build_knowledge_embeddings(
    output_df: pd.DataFrame,
    top_k: int = KNOWLEDGE_TOP_K,
    embedding_model=None,
    normalize_embeddings: bool = NORMALIZE_KNOWLEDGE_EMBEDDINGS
):
    """
    Select overall top-k contexts and return:
        knowledge_embeddings: np.ndarray of shape (top_k, embedding_dim)
        knowledge_mask: np.ndarray of shape (top_k,)

    If fewer than top_k real contexts exist:
        pad with zero vectors.

    If more than top_k real contexts exist:
        keep top_k by final_score.
    """

    if embedding_model is None:
        raise ValueError("embedding_model must be provided.")

    selected_df = select_overall_top_k_from_output_df(
        output_df,
        top_k=top_k
    ).copy()

    # Newer versions use get_embedding_dimension(), while older ones use
    # get_sentence_embedding_dimension().
    if hasattr(embedding_model, "get_embedding_dimension"):
        embedding_dim = embedding_model.get_embedding_dimension()
    else:
        embedding_dim = embedding_model.get_sentence_embedding_dimension()

    real_contexts = selected_df["sentence"].astype(str).tolist()
    num_real = len(real_contexts)

    knowledge_embeddings = np.zeros(
        (top_k, embedding_dim),
        dtype=np.float32
    )

    knowledge_mask = np.zeros(
        (top_k,),
        dtype=np.int64
    )

    if num_real > 0:
        context_embeddings = embedding_model.encode(
            real_contexts,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings
        ).astype(np.float32)

        knowledge_embeddings[:num_real] = context_embeddings
        knowledge_mask[:num_real] = 1

    # Add padding rows to metadata so selected_df also has top_k rows.
    if num_real < top_k:
        pad_rows = []

        for _ in range(top_k - num_real):
            pad_rows.append({
                "matched_query": None,
                "source": None,
                "sentence": "",
                "final_score": None,
                "context_mask": 0,
            })

        selected_df = pd.concat(
            [selected_df, pd.DataFrame(pad_rows)],
            ignore_index=True
        )

    selected_df = selected_df.head(top_k).reset_index(drop=True)

    return knowledge_embeddings, knowledge_mask, selected_df


# ============================================================
# 13. Final retrieval function
# ============================================================

def retrieve_three_contexts_per_query_term(
    meme_text: str,
    image_caption: str,
    generics_df: pd.DataFrame,

    max_terms: int = MAX_QUERY_TERMS,

    use_wikidata: bool = USE_WIKIDATA,
    wikidata_results_per_term: int = WIKIDATA_RESULTS_PER_TERM,
    use_wikidata_exact_label_filter: bool = USE_WIKIDATA_EXACT_LABEL_FILTER,

    use_wikipedia: bool = USE_WIKIPEDIA,
    wikipedia_results_per_term: int = WIKIPEDIA_RESULTS_PER_TERM,
    wikipedia_sentences_per_page: int = WIKIPEDIA_SENTENCES_PER_PAGE,
    wikipedia_exact_title_match_only: bool = WIKIPEDIA_EXACT_TITLE_MATCH_ONLY,
    wikipedia_max_context_words: int = WIKIPEDIA_MAX_CONTEXT_WORDS,
    wikipedia_language: str = WIKIPEDIA_LANGUAGE,

    use_genericskb: bool = USE_GENERICSKB,
    generics_results_per_term: int = GENERICS_RESULTS_PER_TERM,
    min_generics_score: float = MIN_GENERICS_SCORE,

    use_external_lexicons: bool = USE_EXTERNAL_LEXICONS,
    external_lexicon_df: pd.DataFrame = None,
    external_lexicon_results_per_term: int = EXTERNAL_LEXICON_RESULTS_PER_TERM,
    external_lexicon_min_score: float = EXTERNAL_LEXICON_MIN_SCORE,

    alpha_bm25: float = ALPHA_BM25,
    beta_embedding: float = BETA_EMBEDDING,
    gamma_source: float = GAMMA_SOURCE,

    knowledge_top_k: int = KNOWLEDGE_TOP_K,
    normalize_knowledge_embeddings: bool = NORMALIZE_KNOWLEDGE_EMBEDDINGS,

    verbose: bool = False
):
    """
    Retrieve external knowledge contexts for one meme.

    The function:
    1. Combines meme text and image caption into one query.
    2. Extracts query terms.
    3. Retrieves candidates from GenericsKB and external lexicons.
    4. Scores candidates with BM25 and embedding similarity.
    5. Selects one context per enabled source per query term.
    6. Builds the final top-k knowledge embeddings and mask.
    """
    query = f"{meme_text} {image_caption}"

    query_terms = extract_query_terms(
        meme_text=meme_text,
        image_caption=image_caption,
        max_terms=max_terms
    )

    all_candidates = []

    wikidata_candidates = []
    if use_wikidata:
        wikidata_candidates = collect_wikidata_candidates(
            query_terms=query_terms,
            results_per_term=wikidata_results_per_term,
            use_exact_label_filter=use_wikidata_exact_label_filter
        )
        all_candidates.extend(wikidata_candidates)

    wikipedia_candidates = []
    if use_wikipedia:
        wikipedia_candidates = collect_wikipedia_candidates(
            query_terms=query_terms,
            search_results_per_term=wikipedia_results_per_term,
            summary_sentences_per_page=wikipedia_sentences_per_page,
            max_context_words=wikipedia_max_context_words,
            exact_title_match_only=wikipedia_exact_title_match_only,
            language=wikipedia_language
        )
        all_candidates.extend(wikipedia_candidates)

    generics_candidates = []
    if use_genericskb:
        generics_candidates = collect_genericskb_candidates(
            query_terms=query_terms,
            generics_df=generics_df,
            results_per_term=generics_results_per_term,
            min_score=min_generics_score
        )
        all_candidates.extend(generics_candidates)

    external_lexicon_candidates = []
    lexicon_query_terms = []
    if use_external_lexicons and EXTERNAL_LEXICON_AVAILABLE:
        if external_lexicon_df is None:
            external_lexicon_df = globals().get("external_lexicon_df")

        # Use both extracted terms and raw n-grams. This matters because the
        # normal query-term extractor intentionally filters words like "them"
        # and some identity terms, but those are useful for lexicon matching.
        lexicon_query_terms = build_query_terms_for_lexicons(
            query_terms=query_terms,
            full_query=query,
            max_n=4,
        )

        external_lexicon_candidates = collect_external_lexicon_candidates_indexed(
            query_terms=lexicon_query_terms,
            lexicon_index=_external_lexicon_index,
            results_per_term=external_lexicon_results_per_term,
            min_score=external_lexicon_min_score,
        )
        all_candidates.extend(external_lexicon_candidates)

    bm25_candidates = add_bm25_scores(
        query=query,
        candidates=all_candidates
    )

    embedding_candidates = add_embedding_scores(
        query=query,
        candidates=bm25_candidates
    )

    ranked_candidates = combine_scores(
        embedding_candidates,
        alpha_bm25=alpha_bm25,
        beta_embedding=beta_embedding,
        gamma_source=gamma_source
    )

    enabled_sources = []
    if use_wikidata:
        enabled_sources.append("wikidata")
    if use_wikipedia:
        enabled_sources.append("wikipedia")
    if use_genericskb:
        enabled_sources.append("genericskb")

    # Keep the original per-source selection for GenericsKB.
    # External lexicon candidates may come from raw n-grams that are not in
    # query_terms, so we append the real ranked lexicon rows separately.
    per_query_source_rows = select_top_context_per_source_per_query(
        ranked_candidates=ranked_candidates,
        query_terms=query_terms,
        sources=tuple(enabled_sources)
    )

    if use_external_lexicons:
        for item in ranked_candidates:
            if item.get("source") == "external_lexicon":
                real_item = item.copy()
                real_item["context_mask"] = 1
                per_query_source_rows.append(real_item)

    output_df = pd.DataFrame(per_query_source_rows)
    if output_df.empty:
        output_df = pd.DataFrame(columns=[
            "matched_query", "source", "sentence", "final_score", "bm25_score",
            "bm25_score_norm", "embedding_score", "embedding_score_norm",
            "source_score", "source_score_norm", "label_or_term", "qid", "url",
            "context_mask"
        ])

    ordered_cols = [
        "matched_query",
        "source",
        "sentence",
        "final_score",
        "bm25_score",
        "bm25_score_norm",
        "embedding_score",
        "embedding_score_norm",
        "source_score",
        "source_score_norm",
        "label_or_term",
        "qid",
        "url",
        "context_mask",
    ]

    ordered_cols = [col for col in ordered_cols if col in output_df.columns]
    output_df = output_df[ordered_cols]

    knowledge_embeddings, knowledge_mask, selected_contexts_df = build_knowledge_embeddings(
        output_df=output_df,
        top_k=knowledge_top_k,
        embedding_model=embedding_model,
        normalize_embeddings=normalize_knowledge_embeddings
    )

    debug = {
        "query": query,
        "query_terms": query_terms,
        "lexicon_query_terms": lexicon_query_terms,
        "num_wikidata_candidates": len(wikidata_candidates),
        "num_wikipedia_candidates": len(wikipedia_candidates),
        "num_genericskb_candidates": len(generics_candidates),
        "num_external_lexicon_candidates": len(external_lexicon_candidates),
        "num_total_candidates": len(all_candidates),
        "wikidata_candidates": wikidata_candidates,
        "wikipedia_candidates": wikipedia_candidates,
        "genericskb_candidates": generics_candidates,
        "external_lexicon_candidates": external_lexicon_candidates,
        "ranked_candidates": ranked_candidates,
        "contexts_per_query_df": output_df,
        "selected_contexts_df": selected_contexts_df,
        "knowledge_embeddings_shape": knowledge_embeddings.shape,
        "knowledge_mask": knowledge_mask,
    }

    if verbose:
        print("\n================ QUERY ================")
        print(query)

        print("\n================ EXTRACTED QUERY TERMS ================")
        for term in query_terms:
            print("-", term)

        print("\n================ CANDIDATE COUNTS ================")
        print("Wikidata candidates:", len(wikidata_candidates))
        print("Wikipedia candidates:", len(wikipedia_candidates))
        print("GenericsKB candidates:", len(generics_candidates))
        print("External lexicon candidates:", len(external_lexicon_candidates))
        print("Total candidates in shared pool:", len(all_candidates))

        print("\nknowledge_embeddings shape:", knowledge_embeddings.shape)
        print("knowledge_mask:", knowledge_mask)

    return {
        "contexts_per_query_df": output_df,
        "selected_contexts_df": selected_contexts_df,
        "knowledge_embeddings": knowledge_embeddings,
        "knowledge_mask": knowledge_mask,
        "debug": debug,
    }


# ============================================================
# 14. Batch helpers for fast precomputation
# ============================================================

def batch_generate_image_captions(images, batch_size=16):
    captions = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = caption_processor(images=batch, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            ids = caption_model.generate(**inputs, max_new_tokens=CAPTION_MAX_NEW_TOKENS)
        captions.extend(caption_processor.batch_decode(ids, skip_special_tokens=True))
    return captions


def batch_image_to_pixel_values(images, batch_size=32):
    result = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        encoded = blip2_processor(images=batch, return_tensors="pt")
        pv = encoded["pixel_values"]
        for j in range(pv.shape[0]):
            result.append(pv[j])
    return result


def batch_tokenize_texts(texts, max_length=MAX_TEXT_LENGTH, batch_size=64):
    all_ids, all_mask = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = text_tokenizer(
            batch, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        for j in range(enc["input_ids"].shape[0]):
            all_ids.append(enc["input_ids"][j])
            all_mask.append(enc["attention_mask"][j])
    return all_ids, all_mask


def batch_text_to_clip_embeddings(texts, batch_size=64):
    if not USE_CLIP_TEXT_EMBEDDINGS or clip_tokenizer is None or clip_text_model is None:
        return [torch.zeros(CLIP_TEXT_DIM, dtype=torch.float32) for _ in texts]
    result = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = clip_tokenizer(
            batch, padding="max_length", truncation=True,
            max_length=CLIP_MAX_TEXT_LENGTH, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            out = clip_text_model(**enc)
            emb = F.normalize(out.pooler_output.float(), p=2, dim=-1)
        for j in range(emb.shape[0]):
            result.append(emb[j].cpu())
    return result


# ============================================================
# 15. Final function to use for one dataset row
# ============================================================

def process_dataset_row(row):
    """
    Process one row from the meme dataset.

    Expected row format:
        {
            "id": "fhm_42953",
            "image_path": "images/fhm_42953.png",
            "text": "its their character not their color that matters",
            "label": 0,
            "source": "fhm"
        }

    This function returns final model inputs compatible with train.py:
        pixel_values           torch.FloatTensor, shape [3, H, W]
        input_ids              torch.LongTensor,  shape [MAX_TEXT_LENGTH]
        attention_mask         torch.LongTensor,  shape [MAX_TEXT_LENGTH]
        clip_text_embedding    torch.FloatTensor, shape [CLIP_TEXT_DIM]
        knowledge_embeddings   torch.FloatTensor, shape [KNOWLEDGE_TOP_K, 384]
        knowledge_mask         torch.BoolTensor,  shape [KNOWLEDGE_TOP_K]
        labels                 torch.LongTensor scalar
    """

    # Get the image path from the row and join it with the data root.
    image_path = DATA_ROOT / row["image_path"]

    # The dataset text is treated as the OCR/meme text.
    meme_text = row["text"]

    # The label is returned with the model inputs.
    label = row["label"]

    # First, process the image and meme text.
    sample = process_meme_sample(
        image_path=image_path,
        meme_text=meme_text,
        max_text_length=MAX_TEXT_LENGTH
    )

    # Then, retrieve external knowledge using the meme text and generated caption.
    retrieval_result = retrieve_three_contexts_per_query_term(
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

        use_external_lexicons=USE_EXTERNAL_LEXICONS,
        external_lexicon_df=external_lexicon_df,
        external_lexicon_results_per_term=EXTERNAL_LEXICON_RESULTS_PER_TERM,
        external_lexicon_min_score=EXTERNAL_LEXICON_MIN_SCORE,

        alpha_bm25=ALPHA_BM25,
        beta_embedding=BETA_EMBEDDING,
        gamma_source=GAMMA_SOURCE,

        knowledge_top_k=KNOWLEDGE_TOP_K,
        normalize_knowledge_embeddings=NORMALIZE_KNOWLEDGE_EMBEDDINGS,

        verbose=False
    )

    # ------------------------------------------------------------------
    # Compatibility fixes for knowledge_gated.py and train.py
    # ------------------------------------------------------------------
    # process_meme_sample returns tensors with a leading singleton batch
    # dimension, e.g. [1, 3, H, W] and [1, L]. For Dataset.__getitem__, each
    # sample should not include that leading batch dimension, because DataLoader
    # will add the true batch dimension later.
    #
    # build_knowledge_embeddings returns NumPy arrays, but train.py expects
    # tensors so move_batch_to_device can send them to CPU/GPU.
    #
    # train.py expects the key name "labels", not "label".
    # ------------------------------------------------------------------

    return {
        "pixel_values": sample["pixel_values"].squeeze(0).float(),
        "input_ids": sample["input_ids"].squeeze(0).long(),
        "attention_mask": sample["attention_mask"].squeeze(0).long(),
        "clip_text_embedding": sample["clip_text_embedding"].squeeze(0).float(),
        "knowledge_embeddings": torch.tensor(
            retrieval_result["knowledge_embeddings"],
            dtype=torch.float32,
        ),
        "knowledge_mask": torch.tensor(
            retrieval_result["knowledge_mask"],
            dtype=torch.bool,
        ),
        "labels": torch.tensor(label, dtype=torch.long),
    }


# ============================================================
# 15. Local test
# ============================================================
# This part only runs if I execute this file directly.
# It does not run when the file is imported from another training script.

if __name__ == "__main__":
    example_row = {
        "id": "fhm_42953",
        "image_path": "image.jpg",
        "text": "my sandwich-maker is very slow today",
        "label": 0,
        "source": "fhm",
    }

    output = process_dataset_row(example_row)

    print("pixel_values:", output["pixel_values"].shape)
    print("input_ids:", output["input_ids"].shape)
    print("attention_mask:", output["attention_mask"].shape)
    print("clip_text_embedding:", output["clip_text_embedding"].shape)
    print("knowledge_embeddings:", output["knowledge_embeddings"].shape)
    print("knowledge_mask:", output["knowledge_mask"].shape, output["knowledge_mask"])
    print("labels:", output["labels"].shape, output["labels"])
