"""
input_and_context.py

MUTE-stage preprocessing pipeline.

This script prepares one standardized MUTE row for the MemeBLIP2 training stack:

    image -> BLIP2 pixel_values
    text  -> BLIP2 token ids for compatibility/debugging
    text  -> pretrained CLIP text embedding used by the updated student models
    text + caption -> external knowledge retrieval -> MiniLM knowledge embeddings

Main entry point:
    process_dataset_row(row)

Expected row format:
    {
        "id": "mute_000001",
        "image_path": "images/mute_000001.jpg",
        "text": "...",
        "label": 0 or 1,
        "source": "mute"
    }

Returned tensors:
    pixel_values           [3, H, W]
    input_ids              [MAX_TEXT_LENGTH]
    attention_mask         [MAX_TEXT_LENGTH]
    clip_text_embedding    [CLIP_TEXT_DIM]
    knowledge_embeddings   [KNOWLEDGE_TOP_K, KNOWLEDGE_DIM]
    knowledge_mask         [KNOWLEDGE_TOP_K]
    labels                 scalar
"""

from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import requests
import spacy
import torch
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from transformers import (
    Blip2Processor,
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPTextModel,
    CLIPTokenizer,
)

from config import (
    DATASET_ROOT,
    BLIP2_MODEL_NAME,
    CLIP_MODEL_NAME,
    CLIP_TEXT_DIM,
    USE_CLIP_TEXT_EMBEDDINGS,
    KNOWLEDGE_TOP_K,
    KNOWLEDGE_DIM,
    USE_GENERICSKB,
    USE_WIKIDATA,
    USE_WIKIPEDIA,
    USE_EXTERNAL_LEXICONS,
    EXTERNAL_LEXICON_DIR,
    EXTERNAL_LEXICON_RESULTS_PER_TERM,
    EXTERNAL_LEXICON_MIN_SCORE,
    ALPHA_BM25,
    BETA_EMBEDDING,
    GAMMA_SOURCE,
    NORMALIZE_KNOWLEDGE_EMBEDDINGS,
)

try:
    from external_lexicon_knowledge import (
        build_lexicon_index,
        build_query_terms_for_lexicons,
        collect_external_lexicon_candidates,
        collect_external_lexicon_candidates_indexed,
        load_external_lexicon_knowledge,
    )
    EXTERNAL_LEXICON_AVAILABLE = True
except Exception as _external_lexicon_error:
    build_lexicon_index = None
    build_query_terms_for_lexicons = None
    collect_external_lexicon_candidates = None
    collect_external_lexicon_candidates_indexed = None
    load_external_lexicon_knowledge = None
    EXTERNAL_LEXICON_AVAILABLE = False
    _EXTERNAL_LEXICON_IMPORT_ERROR = _external_lexicon_error


# ============================================================
# 1. Runtime settings
# ============================================================

DATA_ROOT = DATASET_ROOT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
CAPTION_MAX_NEW_TOKENS = 30

BLIP2_PROCESSOR_NAME = BLIP2_MODEL_NAME
MAX_TEXT_LENGTH = 128
CLIP_MAX_TEXT_LENGTH = 77

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

GENERICS_DATASET_NAME = "community-datasets/generics_kb"
GENERICS_DATASET_CONFIG = "generics_kb_best"
GENERICS_DATASET_SPLIT = "train"
GENERICS_RESULTS_PER_TERM = 20
MIN_GENERICS_SCORE = 0.0

MAX_QUERY_TERMS = 12

WIKIDATA_RESULTS_PER_TERM = 5
USE_WIKIDATA_EXACT_LABEL_FILTER = True
WIKIDATA_SLEEP = 1.0
WIKIDATA_MAX_RETRIES = 4

WIKIPEDIA_RESULTS_PER_TERM = 2
WIKIPEDIA_SENTENCES_PER_PAGE = 1
WIKIPEDIA_EXACT_TITLE_MATCH_ONLY = True
WIKIPEDIA_MAX_CONTEXT_WORDS = 25
WIKIPEDIA_LANGUAGE = "en"
WIKIPEDIA_SLEEP = 1.5
WIKIPEDIA_MAX_RETRIES = 4

USER_AGENT = "MemeExternalKnowledgeRetrieval/0.8 MUTE academic project"


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# Environment overrides for quick ablations on the cluster.
USE_GENERICSKB = env_flag("USE_GENERICSKB", USE_GENERICSKB)
USE_WIKIDATA = env_flag("USE_WIKIDATA", USE_WIKIDATA)
USE_WIKIPEDIA = env_flag("USE_WIKIPEDIA", USE_WIKIPEDIA)
USE_EXTERNAL_LEXICONS = env_flag("USE_EXTERNAL_LEXICONS", USE_EXTERNAL_LEXICONS)

EXTERNAL_LEXICON_DIR = Path(os.getenv("EXTERNAL_LEXICON_DIR", str(EXTERNAL_LEXICON_DIR)))
EXTERNAL_LEXICON_RESULTS_PER_TERM = int(os.getenv("EXTERNAL_LEXICON_RESULTS_PER_TERM", str(EXTERNAL_LEXICON_RESULTS_PER_TERM)))
EXTERNAL_LEXICON_MIN_SCORE = float(os.getenv("EXTERNAL_LEXICON_MIN_SCORE", str(EXTERNAL_LEXICON_MIN_SCORE)))


# ============================================================
# 2. Load models/resources once
# ============================================================

caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_NAME)
caption_model = BlipForConditionalGeneration.from_pretrained(
    CAPTION_MODEL_NAME,
    use_safetensors=True,
).to(DEVICE)
caption_model.eval()

blip2_processor = Blip2Processor.from_pretrained(BLIP2_PROCESSOR_NAME)
text_tokenizer = blip2_processor.tokenizer
if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token

if USE_CLIP_TEXT_EMBEDDINGS:
    clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
    clip_text_model = CLIPTextModel.from_pretrained(
        CLIP_MODEL_NAME,
        use_safetensors=True,
    ).to(DEVICE)
    clip_text_model.eval()
else:
    clip_tokenizer = None
    clip_text_model = None

# spaCy is English, so it is mostly useful for English/code-mixed tokens and
# generated English captions. Bengali raw OCR is still handled by raw n-grams in
# the lexicon retrieval branch.
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

if USE_GENERICSKB:
    generics_ds = load_dataset(
        GENERICS_DATASET_NAME,
        GENERICS_DATASET_CONFIG,
        split=GENERICS_DATASET_SPLIT,
    )
    generics_df = generics_ds.to_pandas()
else:
    generics_ds = None
    generics_df = pd.DataFrame(columns=["term", "generic_sentence", "score", "term_norm"])


# ============================================================
# 3. Generic helper functions
# ============================================================

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def normalize_genericskb_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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


if USE_GENERICSKB:
    generics_df = normalize_genericskb_dataframe(generics_df)
    _generics_index: dict = {
        term_norm: group.sort_values("score", ascending=False).reset_index(drop=True)
        for term_norm, group in generics_df.groupby("term_norm")
    }
else:
    print("[Info] GenericsKB disabled for MUTE.")
    _generics_index = {}

if USE_EXTERNAL_LEXICONS and not EXTERNAL_LEXICON_AVAILABLE:
    print(
        "[Warning] USE_EXTERNAL_LEXICONS=True but external_lexicon_knowledge.py "
        f"could not be imported: {_EXTERNAL_LEXICON_IMPORT_ERROR}"
    )

if EXTERNAL_LEXICON_AVAILABLE and USE_EXTERNAL_LEXICONS:
    external_lexicon_df = load_external_lexicon_knowledge(EXTERNAL_LEXICON_DIR, include_builtin=False)
    _external_lexicon_index = build_lexicon_index(external_lexicon_df)
    print(f"[Info] Bengali HurtLex source enabled: {len(external_lexicon_df)} rows from {EXTERNAL_LEXICON_DIR}")
else:
    external_lexicon_df = pd.DataFrame(columns=[
        "term", "aliases", "category", "sentence", "score",
        "source_name", "source_file", "citation_key", "term_norm", "aliases_list",
    ])
    _external_lexicon_index = {}


# ============================================================
# 4. Image and text preprocessing
# ============================================================

def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def generate_image_caption(image, max_new_tokens=CAPTION_MAX_NEW_TOKENS):
    inputs = caption_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generated_ids = caption_model.generate(**inputs, max_new_tokens=max_new_tokens)
    return caption_processor.decode(generated_ids[0], skip_special_tokens=True)


def image_to_pixel_values(image):
    encoded_image = blip2_processor(images=image, return_tensors="pt")
    return encoded_image["pixel_values"]


def tokenize_text(text, max_length=MAX_TEXT_LENGTH):
    encoded_text = text_tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded_text["input_ids"], encoded_text["attention_mask"]


def text_to_clip_embedding(text, max_length=CLIP_MAX_TEXT_LENGTH):
    """
    Convert raw meme OCR/caption text into a pretrained CLIP text embedding.
    This is the text representation consumed by the updated student models.
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
            f"Expected CLIP_TEXT_DIM={CLIP_TEXT_DIM}, got {clip_text_embedding.shape[-1]}."
        )

    return clip_text_embedding.cpu()


def process_meme_sample(image_path, meme_text=None, max_text_length=MAX_TEXT_LENGTH):
    image = load_image(image_path)
    pixel_values = image_to_pixel_values(image)
    image_caption = generate_image_caption(image)

    if meme_text is None:
        raise ValueError("No meme_text provided. MUTE rows should contain text/Captions.")

    ocr_text = str(meme_text)
    input_ids, attention_mask = tokenize_text(ocr_text, max_length=max_text_length)
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
# 5. Query term extraction
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
    "person", "people", "man", "woman", "men", "women", "guy", "girl", "boy",
    "show", "showing", "shown", "stand", "standing", "sit", "sitting",
    "look", "looking", "appear", "appearing", "depict", "depicting", "display", "displaying",
}


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def simple_tokenize(text: str):
    """
    Unicode-aware tokenizer for BM25 over Bengali/code-mixed text.
    """
    return re.findall(
        r"[A-Za-zÀ-ÖØ-öø-ÿ\u0980-\u09FF0-9][A-Za-zÀ-ÖØ-öø-ÿ\u0980-\u09FF0-9\-']*",
        str(text).lower(),
        flags=re.UNICODE,
    )


def unique_preserve_order(items: Iterable[str]):
    output = []
    seen = set()
    for item in items:
        if item is None:
            continue
        key = str(item).lower().strip()
        if key and key not in seen:
            output.append(str(item).strip())
            seen.add(key)
    return output


def strip_edge_connectors(term: str):
    if not term:
        return term

    words = term.strip().split()
    while words and words[0].lower() in {"of", "and", "or", "&"}:
        words = words[1:]
    while words and words[-1].lower() in {"of", "and", "or", "the", "&"}:
        words = words[:-1]
    return " ".join(words)


def clean_term(term: str):
    if not term:
        return None
    term = re.sub(r"\s+", " ", str(term).strip())
    term = strip_edge_connectors(term)
    if len(term) < 3:
        return None
    if term.lower() in GENERIC_BAD_TERMS:
        return None
    return term


def safe_lemma(token):
    lower = token.text.lower()
    if lower in {"media", "news", "police"}:
        return lower
    return token.lemma_.lower()


def is_connector_token(token):
    return token.text.lower() in {"of", "the", "and", "&"}


def extract_title_or_propn_phrases(doc):
    phrases = []
    current = []

    def flush():
        nonlocal current
        if current:
            phrase = clean_term(strip_edge_connectors(" ".join(current)))
            if phrase:
                phrases.append(phrase)
        current = []

    for token in doc:
        if token.is_punct:
            flush()
        elif token.pos_ == "PROPN":
            current.append(token.text)
        elif current and is_connector_token(token):
            current.append(token.text)
        else:
            flush()
    flush()
    return phrases


def remove_subterms(terms):
    terms = unique_preserve_order(terms)
    final_terms = []

    for term in terms:
        term_lower = term.lower()
        is_subterm = False
        for other in terms:
            other_lower = other.lower()
            if term_lower == other_lower:
                continue
            if re.search(r"\b" + re.escape(term_lower) + r"\b", other_lower) and len(other_lower) > len(term_lower):
                is_subterm = True
                break
        if not is_subterm:
            final_terms.append(term)

    return final_terms


def extract_query_terms(
    meme_text: str,
    image_caption: str,
    max_terms: int = MAX_QUERY_TERMS,
    include_pos={"NOUN", "PROPN", "VERB", "ADJ"},
):
    """
    Extract English/code-mixed terms with spaCy plus useful raw Bengali tokens.
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
            if token.is_punct or token.pos_ == "DET":
                continue
            if token.is_stop and token.text.lower() not in {"of", "and"}:
                continue
            if token.pos_ == "PROPN":
                cleaned_tokens.append(token.text)
            elif token.pos_ in {"NOUN", "ADJ"}:
                cleaned_tokens.append(safe_lemma(token))
            elif token.text.lower() in {"of", "and"}:
                cleaned_tokens.append(token.text.lower())

        phrase = clean_term(" ".join(cleaned_tokens))
        if phrase:
            terms.append(phrase)

    for token in doc:
        if token.is_stop or token.is_punct or token.pos_ not in include_pos or len(token.text) < 3:
            continue
        term = token.text if token.pos_ == "PROPN" else safe_lemma(token)
        term = clean_term(term)
        if term:
            terms.append(term)

    # Add raw Unicode tokens because Bengali words are not reliably handled by en_core_web_sm.
    for token in simple_tokenize(meme_text):
        cleaned = clean_term(token)
        if cleaned:
            terms.append(cleaned)

    terms = unique_preserve_order(terms)
    terms = remove_subterms(terms)
    return terms[:max_terms]


# ============================================================
# 6. Wikidata / Wikipedia retrieval
# ============================================================

WIKIDATA_CACHE = {}
WIKIPEDIA_SEARCH_CACHE = {}
WIKIPEDIA_SUMMARY_CACHE = {}


def wikidata_entity_to_sentence(label: str, description: str):
    label = str(label).strip()
    description = re.sub(r"\s+", " ", str(description).strip().rstrip("."))
    if not label or not description:
        return None
    if description.lower().startswith(("a ", "an ", "the ")):
        return f"{label} is {description}."
    return f"{label} is a {description}."


def search_wikidata(term: str, limit=WIKIDATA_RESULTS_PER_TERM, sleep=WIKIDATA_SLEEP, max_retries=WIKIDATA_MAX_RETRIES):
    cache_key = (term.lower().strip(), limit)
    if cache_key in WIKIDATA_CACHE:
        return WIKIDATA_CACHE[cache_key]

    url = "https://www.wikidata.org/w/api.php"
    params = {"action": "wbsearchentities", "search": term, "language": "en", "format": "json", "limit": limit, "type": "item"}
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_time = float(retry_after) if retry_after is not None else sleep * (2 ** attempt) + 2.0
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            results = response.json().get("search", [])
            WIKIDATA_CACHE[cache_key] = results
            time.sleep(sleep)
            return results
        except requests.RequestException:
            time.sleep(sleep * (2 ** attempt) + 2.0)

    WIKIDATA_CACHE[cache_key] = []
    return []


def collect_wikidata_candidates(query_terms, results_per_term=WIKIDATA_RESULTS_PER_TERM, use_exact_label_filter=USE_WIKIDATA_EXACT_LABEL_FILTER):
    candidates = []
    for term in query_terms:
        for result in search_wikidata(term, limit=results_per_term):
            qid = result.get("id")
            label = result.get("label")
            description = result.get("description")
            if not qid or not label or not description:
                continue
            if use_exact_label_filter and label.strip() != term.strip():
                continue
            sentence = wikidata_entity_to_sentence(label, description)
            if sentence:
                candidates.append({
                    "source": "wikidata",
                    "matched_query": term,
                    "label_or_term": label,
                    "sentence": sentence,
                    "source_score": float(result.get("score", 0.0)),
                    "url": f"https://www.wikidata.org/wiki/{qid}",
                    "qid": qid,
                    "raw_description": description,
                })
    return candidates


def search_wikipedia_pages(term: str, limit=WIKIPEDIA_RESULTS_PER_TERM, language=WIKIPEDIA_LANGUAGE, sleep=WIKIPEDIA_SLEEP, max_retries=WIKIPEDIA_MAX_RETRIES):
    cache_key = (language, term.lower().strip(), limit)
    if cache_key in WIKIPEDIA_SEARCH_CACHE:
        return WIKIPEDIA_SEARCH_CACHE[cache_key]

    url = f"https://{language}.wikipedia.org/w/rest.php/v1/search/page"
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params={"q": term, "limit": limit}, headers=headers, timeout=15)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_time = float(retry_after) if retry_after else sleep * (2 ** attempt) + 2.0
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            pages = response.json().get("pages", [])
            WIKIPEDIA_SEARCH_CACHE[cache_key] = pages
            time.sleep(sleep)
            return pages
        except requests.RequestException:
            time.sleep(sleep * (2 ** attempt) + 2.0)

    WIKIPEDIA_SEARCH_CACHE[cache_key] = []
    return []


def get_wikipedia_summary(title: str, language=WIKIPEDIA_LANGUAGE, sleep=WIKIPEDIA_SLEEP, max_retries=WIKIPEDIA_MAX_RETRIES):
    cache_key = (language, title)
    if cache_key in WIKIPEDIA_SUMMARY_CACHE:
        return WIKIPEDIA_SUMMARY_CACHE[cache_key]

    url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 404:
                WIKIPEDIA_SUMMARY_CACHE[cache_key] = None
                return None
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_time = float(retry_after) if retry_after else sleep * (2 ** attempt) + 2.0
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            data = response.json()
            WIKIPEDIA_SUMMARY_CACHE[cache_key] = data
            time.sleep(sleep)
            return data
        except requests.RequestException:
            time.sleep(sleep * (2 ** attempt) + 2.0)

    WIKIPEDIA_SUMMARY_CACHE[cache_key] = None
    return None


def is_bad_wikipedia_sentence(sentence: str) -> bool:
    if not sentence:
        return True
    s = sentence.strip().lower()
    bad_patterns = ["may refer to", "can refer to", "refers to", "is a list of", "list of", "index of", "set index", "outline of", "glossary of"]
    return any(p in s for p in bad_patterns) or len(sentence.split()) < 5


def split_summary_into_sentences(text: str, max_sentences=WIKIPEDIA_SENTENCES_PER_PAGE, max_context_words=WIKIPEDIA_MAX_CONTEXT_WORDS):
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
        if not is_bad_wikipedia_sentence(sent):
            cleaned.append(sent)
        if len(cleaned) >= max_sentences:
            break
    return cleaned


def collect_wikipedia_candidates(query_terms, search_results_per_term=WIKIPEDIA_RESULTS_PER_TERM, summary_sentences_per_page=WIKIPEDIA_SENTENCES_PER_PAGE, max_context_words=WIKIPEDIA_MAX_CONTEXT_WORDS, exact_title_match_only=WIKIPEDIA_EXACT_TITLE_MATCH_ONLY, language=WIKIPEDIA_LANGUAGE):
    candidates = []
    for term in query_terms:
        for rank, page in enumerate(search_wikipedia_pages(term, limit=search_results_per_term, language=language)):
            title = page.get("title")
            if not title:
                continue
            if exact_title_match_only and term.strip().lower() != title.strip().lower():
                continue
            summary_data = get_wikipedia_summary(title=title, language=language)
            if not summary_data or not summary_data.get("extract"):
                continue
            page_url = summary_data.get("content_urls", {}).get("desktop", {}).get("page")
            for sent_idx, sent in enumerate(split_summary_into_sentences(summary_data["extract"], summary_sentences_per_page, max_context_words)):
                candidates.append({
                    "source": "wikipedia",
                    "matched_query": term,
                    "label_or_term": title,
                    "sentence": sent,
                    "source_score": 1.0 / (1.0 + rank + sent_idx),
                    "url": page_url,
                    "qid": None,
                    "raw_description": page.get("description"),
                })
    return candidates


# ============================================================
# 7. GenericsKB retrieval
# ============================================================

def search_genericskb(term: str, generics_df: pd.DataFrame, limit=GENERICS_RESULTS_PER_TERM, min_score=MIN_GENERICS_SCORE):
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
        if sentence:
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


def collect_genericskb_candidates(query_terms, generics_df: pd.DataFrame, results_per_term=GENERICS_RESULTS_PER_TERM, min_score=MIN_GENERICS_SCORE):
    candidates = []
    for term in query_terms:
        candidates.extend(search_genericskb(term, generics_df, limit=results_per_term, min_score=min_score))
    return candidates


# ============================================================
# 8. Scoring / ranking
# ============================================================

def add_bm25_scores(query: str, candidates):
    if not candidates:
        return []
    documents = [item["sentence"] for item in candidates]
    tokenized_docs = [simple_tokenize(document) for document in documents]
    tokenized_query = simple_tokenize(query)
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)
    return [{**item, "bm25_score": float(score)} for item, score in zip(candidates, scores)]


def add_embedding_scores(query: str, candidates, batch_size: int = 32):
    if not candidates:
        return []
    candidate_sentences = [item["sentence"] for item in candidates]
    query_embedding = embedding_model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    candidate_embeddings = embedding_model.encode(candidate_sentences, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size)
    scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
    return [{**item, "embedding_score": float(score)} for item, score in zip(candidates, scores)]


def minmax_normalize(values):
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return values
    min_v = values.min()
    max_v = values.max()
    if math.isclose(min_v, max_v):
        return np.zeros_like(values)
    return (values - min_v) / (max_v - min_v)


def add_normalized_source_scores_by_source(candidates):
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
    return [{**item, "source_score_norm": float(norm_scores[i])} for i, item in enumerate(candidates)]


def combine_scores(candidates, alpha_bm25=ALPHA_BM25, beta_embedding=BETA_EMBEDDING, gamma_source=GAMMA_SOURCE):
    if not candidates:
        return []
    candidates = add_normalized_source_scores_by_source(candidates)
    bm25_norm = minmax_normalize([item.get("bm25_score", 0.0) for item in candidates])
    emb_norm = minmax_normalize([item.get("embedding_score", 0.0) for item in candidates])

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
# 9. Context selection and embedding
# ============================================================

def empty_context_for_query_source(query_term: str, source: str):
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


def select_top_context_per_source_per_query(ranked_candidates, query_terms, sources=("wikidata", "wikipedia", "genericskb")):
    rows = []
    for query_term in query_terms:
        for source in sources:
            matching = [
                item for item in ranked_candidates
                if item.get("matched_query") == query_term and item.get("source") == source
            ]
            if matching:
                best = matching[0].copy()
                best["context_mask"] = 1
                rows.append(best)
            else:
                rows.append(empty_context_for_query_source(query_term, source))
    return rows


def select_overall_top_k_from_output_df(output_df, top_k: int = KNOWLEDGE_TOP_K):
    real_df = output_df[
        (output_df["context_mask"] == 1)
        & (output_df["final_score"].notna())
        & (output_df["sentence"].astype(str).str.len() > 0)
    ].copy()
    real_df = real_df.sort_values("final_score", ascending=False)
    return real_df.head(top_k)


def build_knowledge_embeddings(output_df: pd.DataFrame, top_k: int = KNOWLEDGE_TOP_K, embedding_model=None, normalize_embeddings: bool = NORMALIZE_KNOWLEDGE_EMBEDDINGS):
    if embedding_model is None:
        raise ValueError("embedding_model must be provided.")

    selected_df = select_overall_top_k_from_output_df(output_df, top_k=top_k).copy()

    if hasattr(embedding_model, "get_embedding_dimension"):
        embedding_dim = embedding_model.get_embedding_dimension()
    else:
        embedding_dim = embedding_model.get_sentence_embedding_dimension()

    if embedding_dim != KNOWLEDGE_DIM:
        raise ValueError(f"Expected KNOWLEDGE_DIM={KNOWLEDGE_DIM}, got embedding_dim={embedding_dim}.")

    real_contexts = selected_df["sentence"].astype(str).tolist()
    num_real = len(real_contexts)

    knowledge_embeddings = np.zeros((top_k, embedding_dim), dtype=np.float32)
    knowledge_mask = np.zeros((top_k,), dtype=np.int64)

    if num_real > 0:
        context_embeddings = embedding_model.encode(
            real_contexts,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        ).astype(np.float32)
        knowledge_embeddings[:num_real] = context_embeddings
        knowledge_mask[:num_real] = 1

    if num_real < top_k:
        pad_rows = [{"matched_query": None, "source": None, "sentence": "", "final_score": None, "context_mask": 0} for _ in range(top_k - num_real)]
        selected_df = pd.concat([selected_df, pd.DataFrame(pad_rows)], ignore_index=True)

    selected_df = selected_df.head(top_k).reset_index(drop=True)
    return knowledge_embeddings, knowledge_mask, selected_df


# ============================================================
# 10. Main retrieval function
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
    verbose: bool = False,
):
    query = f"{meme_text or ''} {image_caption or ''}".strip()
    query_terms = extract_query_terms(meme_text=meme_text, image_caption=image_caption, max_terms=max_terms)

    all_candidates = []

    wikidata_candidates = []
    if use_wikidata:
        wikidata_candidates = collect_wikidata_candidates(query_terms, wikidata_results_per_term, use_wikidata_exact_label_filter)
        all_candidates.extend(wikidata_candidates)

    wikipedia_candidates = []
    if use_wikipedia:
        wikipedia_candidates = collect_wikipedia_candidates(
            query_terms,
            search_results_per_term=wikipedia_results_per_term,
            summary_sentences_per_page=wikipedia_sentences_per_page,
            max_context_words=wikipedia_max_context_words,
            exact_title_match_only=wikipedia_exact_title_match_only,
            language=wikipedia_language,
        )
        all_candidates.extend(wikipedia_candidates)

    generics_candidates = []
    if use_genericskb:
        generics_candidates = collect_genericskb_candidates(query_terms, generics_df, generics_results_per_term, min_generics_score)
        all_candidates.extend(generics_candidates)

    external_lexicon_candidates = []
    lexicon_query_terms = []
    if use_external_lexicons and EXTERNAL_LEXICON_AVAILABLE:
        if external_lexicon_df is None:
            external_lexicon_df = globals().get("external_lexicon_df")

        lexicon_query_terms = build_query_terms_for_lexicons(
            query_terms=query_terms,
            full_query=query,
            max_n=4,
        )

        if collect_external_lexicon_candidates_indexed is not None and _external_lexicon_index:
            external_lexicon_candidates = collect_external_lexicon_candidates_indexed(
                query_terms=lexicon_query_terms,
                lexicon_index=_external_lexicon_index,
                results_per_term=external_lexicon_results_per_term,
                min_score=external_lexicon_min_score,
            )
        else:
            external_lexicon_candidates = collect_external_lexicon_candidates(
                query_terms=lexicon_query_terms,
                lexicon_df=external_lexicon_df,
                results_per_term=external_lexicon_results_per_term,
                min_score=external_lexicon_min_score,
            )

        all_candidates.extend(external_lexicon_candidates)

    ranked_candidates = combine_scores(
        add_embedding_scores(query, add_bm25_scores(query, all_candidates)),
        alpha_bm25=alpha_bm25,
        beta_embedding=beta_embedding,
        gamma_source=gamma_source,
    )

    enabled_sources = []
    if use_wikidata:
        enabled_sources.append("wikidata")
    if use_wikipedia:
        enabled_sources.append("wikipedia")
    if use_genericskb:
        enabled_sources.append("genericskb")

    per_query_source_rows = select_top_context_per_source_per_query(
        ranked_candidates=ranked_candidates,
        query_terms=query_terms,
        sources=tuple(enabled_sources),
    )

    # External lexicon candidates come from raw n-grams that may not be in query_terms.
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
            "context_mask",
        ])

    ordered_cols = [
        "matched_query", "source", "sentence", "final_score", "bm25_score",
        "bm25_score_norm", "embedding_score", "embedding_score_norm",
        "source_score", "source_score_norm", "label_or_term", "qid", "url", "context_mask",
    ]
    output_df = output_df[[col for col in ordered_cols if col in output_df.columns]]

    knowledge_embeddings, knowledge_mask, selected_contexts_df = build_knowledge_embeddings(
        output_df=output_df,
        top_k=knowledge_top_k,
        embedding_model=embedding_model,
        normalize_embeddings=normalize_knowledge_embeddings,
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
        print("Total candidates:", len(all_candidates))
        print("knowledge_embeddings shape:", knowledge_embeddings.shape)
        print("knowledge_mask:", knowledge_mask)

    return {
        "contexts_per_query_df": output_df,
        "selected_contexts_df": selected_contexts_df,
        "knowledge_embeddings": knowledge_embeddings,
        "knowledge_mask": knowledge_mask,
        "debug": debug,
    }


# ============================================================
# 11. Batch helpers for precomputation
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
        enc = text_tokenizer(batch, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
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
        enc = clip_tokenizer(batch, padding="max_length", truncation=True, max_length=CLIP_MAX_TEXT_LENGTH, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = clip_text_model(**enc)
            emb = F.normalize(out.pooler_output.float(), p=2, dim=-1)
        for j in range(emb.shape[0]):
            result.append(emb[j].cpu())
    return result


# ============================================================
# 12. Final row processing function
# ============================================================

def process_dataset_row(row):
    image_path = DATA_ROOT / row["image_path"]
    meme_text = row["text"]
    label = row["label"]

    sample = process_meme_sample(
        image_path=image_path,
        meme_text=meme_text,
        max_text_length=MAX_TEXT_LENGTH,
    )

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
        verbose=False,
    )

    return {
        "pixel_values": sample["pixel_values"].squeeze(0).float(),
        "input_ids": sample["input_ids"].squeeze(0).long(),
        "attention_mask": sample["attention_mask"].squeeze(0).long(),
        "clip_text_embedding": sample["clip_text_embedding"].squeeze(0).float(),
        "knowledge_embeddings": torch.tensor(retrieval_result["knowledge_embeddings"], dtype=torch.float32),
        "knowledge_mask": torch.tensor(retrieval_result["knowledge_mask"], dtype=torch.bool),
        "labels": torch.tensor(int(label), dtype=torch.long),
    }


# ============================================================
# 13. Optional local test
# ============================================================

if __name__ == "__main__":
    example_row = {
        "id": "mute_000000",
        "image_path": "images/example.jpg",
        "text": "বাংলা মিম টেক্সট",
        "label": 0,
        "source": "mute",
    }

    output = process_dataset_row(example_row)
    print("pixel_values:", output["pixel_values"].shape)
    print("input_ids:", output["input_ids"].shape)
    print("attention_mask:", output["attention_mask"].shape)
    print("clip_text_embedding:", output["clip_text_embedding"].shape)
    print("knowledge_embeddings:", output["knowledge_embeddings"].shape)
    print("knowledge_mask:", output["knowledge_mask"].shape, output["knowledge_mask"])
    print("labels:", output["labels"].shape, output["labels"])
