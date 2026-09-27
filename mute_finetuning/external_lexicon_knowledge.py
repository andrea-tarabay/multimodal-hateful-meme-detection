"""
external_lexicon_knowledge.py

Bengali/Bangla-only external lexical knowledge retrieval for the MUTE stage.

This module intentionally loads ONLY Bengali HurtLex:

    external_lexicons_bn_only/hurtlex_BN.tsv

It does NOT load:
    - English HurtLex/Plaza lexicons
    - MOL
    - built-in English social cue templates
    - external_lexicons_combined.csv

The returned candidates are lexical cues, not labels. A matched HurtLex term only
means: inspect whether this OCR span is actually used offensively, vulgarly,
insultingly, objectifyingly, stereotypingly, dehumanizingly, or degradingly in
image-text context.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    from config import EXTERNAL_LEXICON_DIR
except Exception:
    EXTERNAL_LEXICON_DIR = Path(__file__).resolve().parent / "external_lexicons_bn_only"


# ============================================================
# 1. Metadata / citations
# ============================================================

CITATIONS = {
    "bassignana2018hurtlex": {
        "short": "Bassignana et al. (2018)",
        "bibtex": """@inproceedings{bassignana2018hurtlex,
  title={Hurtlex: A multilingual lexicon of words to hurt},
  author={Bassignana, Elisa and Basile, Valerio and Patti, Viviana},
  booktitle={Proceedings of the 5th Italian Conference on Computational Linguistics},
  pages={1--6},
  year={2018}
}""",
    },
}


# ============================================================
# 2. HurtLex category metadata
# ============================================================

HURTLEX_CATEGORY_DESCRIPTIONS = {
    "ps": "negative stereotypes and ethnic or identity-related slurs",
    "rci": "locations and demonyms",
    "pa": "professions and occupations used as insults",
    "ddp": "physical disabilities and diversity",
    "ddf": "cognitive disabilities and diversity",
    "dmc": "moral or behavioral defects",
    "is": "words related to social and economic disadvantage",
    "or": "offensive or derogatory terms",
    "an": "animal names that may be used for insulting or dehumanizing comparisons",
    "asm": "male genitalia/body-related offensive terms",
    "asf": "female genitalia/body-related offensive terms",
    "pr": "prostitution-related offensive terms",
    "om": "homosexuality-related offensive terms",
    "qas": "words connected to alleged moral weakness or stupidity",
    "cds": "derogatory words connected to social behavior",
    "re": "felonies and crime-related terms",
    "svp": "words connected to vices, sins, or negative social positioning",
}

HURTLEX_CATEGORY_PRIORITY = {
    "or": 100,
    "ps": 95,
    "qas": 90,
    "cds": 85,
    "dmc": 80,
    "ddf": 78,
    "ddp": 78,
    "om": 75,
    "asf": 70,
    "asm": 70,
    "re": 68,
    "pa": 65,
    "is": 60,
    "an": 55,
    "svp": 50,
    "rci": 45,
}

BENGALI_HURTLEX_FILENAMES = [
    "hurtlex_BN.tsv",
    "hurtlex_bn.tsv",
]


# ============================================================
# 3. Text utilities
# ============================================================

def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def simple_tokenize(text: str) -> List[str]:
    """
    Unicode-aware tokenizer for Bengali/Bangla and code-mixed OCR text.

    Keeps:
        - Bengali block U+0980-U+09FF
        - Latin letters for code-mixed text
        - digits, apostrophes, hyphens
    """
    text = str(text).lower()
    return re.findall(
        r"[A-Za-zÀ-ÖØ-öø-ÿ\u0980-\u09FF0-9][A-Za-zÀ-ÖØ-öø-ÿ\u0980-\u09FF0-9\-']*",
        text,
        flags=re.UNICODE,
    )


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    output: List[str] = []
    seen = set()

    for item in items:
        if item is None:
            continue
        value = str(item).strip()
        if not value:
            continue
        key = normalize_text(value)
        if key not in seen:
            output.append(value)
            seen.add(key)

    return output


def split_aliases(value) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = str(value).split(";")
    return [str(x).strip() for x in raw_items if str(x).strip()]


def make_ngrams_from_query(query: str, max_n: int = 4) -> List[str]:
    tokens = simple_tokenize(query)
    phrases: List[str] = []

    for n in range(1, max_n + 1):
        for i in range(0, max(0, len(tokens) - n + 1)):
            phrases.append(" ".join(tokens[i:i + n]))

    return unique_preserve_order(phrases)


def build_query_terms_for_lexicons(
    query_terms: Iterable[str],
    full_query: str,
    max_n: int = 4,
) -> List[str]:
    """
    Use both extracted terms and raw OCR n-grams for lexical matching.
    This is important for Bengali OCR because spaCy English extraction can miss
    Bengali tokens entirely.
    """
    return unique_preserve_order(list(query_terms) + make_ngrams_from_query(full_query, max_n=max_n))


# ============================================================
# 4. Loading Bengali HurtLex
# ============================================================

def normalize_external_lexicon_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    defaults = {
        "term": "",
        "aliases": "",
        "category": "hurtlex_bn",
        "sentence": "Bengali HurtLex lexical cue matched; inspect the meme context before using it as evidence.",
        "score": 0.75,
        "source_name": "HurtLex_BN",
        "source_file": "hurtlex_BN.tsv",
        "citation_key": "bassignana2018hurtlex",
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    for col in ["term", "aliases", "category", "sentence", "source_name", "source_file", "citation_key"]:
        df[col] = df[col].fillna("").astype(str)

    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.75)
    df["term_norm"] = df["term"].apply(normalize_text)
    df["aliases_list"] = df["aliases"].apply(split_aliases)

    df = df[df["term_norm"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["term_norm", "source_name", "category"]).reset_index(drop=True)
    return df


def load_hurtlex_tsv(path: Path, language: str = "BN") -> pd.DataFrame:
    """
    Load a Bengali HurtLex TSV file into the pipeline's external-knowledge format.

    Expected HurtLex columns include:
        id, pos, category, stereotype, lemma, level
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as error:
        print(f"[Warning] Could not load HurtLex TSV {path}: {error}")
        return pd.DataFrame()

    if "lemma" not in df.columns:
        print(f"[Warning] HurtLex TSV missing required 'lemma' column: {path}")
        return pd.DataFrame()

    rows: List[Dict] = []

    for _, row in df.iterrows():
        term = str(row.get("lemma", "")).strip()
        if not term or term.lower() in {"nan", "none", "null"}:
            continue

        category_code = str(row.get("category", "hurtlex")).strip().lower() or "hurtlex"
        category_desc = HURTLEX_CATEGORY_DESCRIPTIONS.get(category_code, "HurtLex lexical cue")
        level = str(row.get("level", "inclusive")).strip().lower() or "inclusive"
        stereotype = str(row.get("stereotype", "no")).strip().lower() or "no"
        row_id = str(row.get("id", "")).strip()
        pos = str(row.get("pos", "")).strip()

        if level == "conservative":
            score = 0.88
        elif stereotype == "yes":
            score = 0.80
        else:
            score = 0.68

        sentence = (
            f"Bengali HurtLex term matched in category '{category_code}' "
            f"({category_desc}). Use this only as lexical context: check whether the OCR term "
            "is actually used as offensive, vulgar, insulting, objectifying, stereotyping, "
            "dehumanizing, or degrading content in the meme."
        )

        rows.append({
            "term": term,
            "aliases": "",
            "category": f"hurtlex_{language.lower()}_{category_code}",
            "sentence": sentence,
            "score": float(score),
            "source_name": f"HurtLex_{language.upper()}",
            "source_file": path.name,
            "citation_key": "bassignana2018hurtlex",
            "hurtlex_id": row_id,
            "hurtlex_pos": pos,
            "hurtlex_level": level,
            "hurtlex_stereotype": stereotype,
            "hurtlex_category_code": category_code,
            "hurtlex_category_description": category_desc,
            "hurtlex_category_priority": HURTLEX_CATEGORY_PRIORITY.get(category_code, 0),
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["score", "hurtlex_category_priority", "category", "term"],
        ascending=[False, False, True, True],
    )
    out = out.drop_duplicates(subset=["term", "source_name"], keep="first")
    out = out.drop(columns=["hurtlex_category_priority"], errors="ignore")
    return out.reset_index(drop=True)


def load_external_lexicon_knowledge(
    lexicon_dir: Optional[Path] = None,
    include_builtin: bool = False,
) -> pd.DataFrame:
    """
    Load Bengali HurtLex rows only.

    include_builtin is accepted for compatibility with first-stage code but is
    ignored. This function never loads built-in, English, MOL, or combined CSV
    lexicons.
    """
    if lexicon_dir is None:
        lexicon_dir = EXTERNAL_LEXICON_DIR

    lexicon_dir = Path(lexicon_dir)
    frames: List[pd.DataFrame] = []

    for filename in BENGALI_HURTLEX_FILENAMES:
        path = lexicon_dir / filename
        if path.exists():
            frames.append(load_hurtlex_tsv(path, language="BN"))

    if not frames:
        return normalize_external_lexicon_dataframe(pd.DataFrame())

    combined = pd.concat(frames, ignore_index=True)
    return normalize_external_lexicon_dataframe(combined)


# ============================================================
# 5. Matching and candidate collection
# ============================================================

def phrase_match_score(lexicon_phrase: str, query_term: str) -> float:
    phrase = normalize_text(lexicon_phrase)
    term = normalize_text(query_term)

    if not phrase or not term:
        return 0.0

    if phrase == term:
        return 1.0

    phrase_tokens = simple_tokenize(phrase)
    term_tokens = simple_tokenize(term)

    if not phrase_tokens or not term_tokens:
        return 0.0

    if len(phrase_tokens) > 1:
        n = len(phrase_tokens)
        for i in range(0, len(term_tokens) - n + 1):
            if term_tokens[i:i + n] == phrase_tokens:
                return 0.95
        return 0.0

    if phrase_tokens[0] in set(term_tokens):
        return 0.90

    return 0.0


def search_external_lexicons(
    term: str,
    lexicon_df: pd.DataFrame,
    limit: int = 20,
    min_score: float = 0.55,
) -> List[Dict]:
    if lexicon_df is None or len(lexicon_df) == 0:
        return []

    candidates: List[Dict] = []

    for _, row in lexicon_df.iterrows():
        lexicon_term = str(row.get("term", "")).strip()
        phrases = [lexicon_term] + split_aliases(row.get("aliases", ""))

        match = max(phrase_match_score(p, term) for p in phrases)
        if match < min_score:
            continue

        source_score = float(row.get("score", 0.75)) * float(match)

        candidates.append({
            "source": "external_lexicon",
            "matched_query": term,
            "matched_lexicon_term": lexicon_term,
            "label_or_term": f"{row.get('source_name', 'HurtLex_BN')}::{row.get('category', 'hurtlex_bn')}",
            "sentence": row.get("sentence", "Bengali HurtLex lexical cue matched; inspect context."),
            "source_score": source_score,
            "url": None,
            "qid": None,
            "raw_description": (
                f"source_file={row.get('source_file', '')}; "
                f"citation={row.get('citation_key', '')}; "
                f"hurtlex_id={row.get('hurtlex_id', '')}; "
                f"level={row.get('hurtlex_level', '')}; "
                f"stereotype={row.get('hurtlex_stereotype', '')}; "
                f"category={row.get('hurtlex_category_code', '')}"
            ),
        })

    candidates.sort(key=lambda x: x["source_score"], reverse=True)
    return candidates[:limit]


def collect_external_lexicon_candidates(
    query_terms: Iterable[str],
    lexicon_df: pd.DataFrame,
    results_per_term: int = 20,
    min_score: float = 0.55,
) -> List[Dict]:
    """
    Collect candidates and deduplicate by actual HurtLex lemma/category rather
    than by OCR n-gram.
    """
    all_candidates: List[Dict] = []

    for term in query_terms:
        all_candidates.extend(
            search_external_lexicons(
                term=term,
                lexicon_df=lexicon_df,
                limit=results_per_term,
                min_score=min_score,
            )
        )

    best_by_key: Dict = {}

    for item in all_candidates:
        key = (
            normalize_text(item.get("matched_lexicon_term", item.get("matched_query", ""))),
            item.get("label_or_term", ""),
        )

        old = best_by_key.get(key)
        if old is None or item.get("source_score", 0.0) > old.get("source_score", 0.0):
            best_by_key[key] = item

    deduped = list(best_by_key.values())
    deduped.sort(key=lambda x: x.get("source_score", 0.0), reverse=True)
    return deduped


def build_lexicon_index(lexicon_df: pd.DataFrame) -> dict:
    """
    Build {phrase_norm: [row_dicts]} for direct lookup.

    This mirrors the first-stage optimized API. It is optional; the non-indexed
    collector remains available for backward compatibility.
    """
    index: dict = {}

    if lexicon_df is None or len(lexicon_df) == 0:
        return index

    for _, row in lexicon_df.iterrows():
        row_dict = row.to_dict()
        aliases = split_aliases(row_dict.get("aliases", ""))
        for phrase in [row_dict.get("term", "")] + aliases:
            key = normalize_text(phrase)
            if key:
                index.setdefault(key, []).append(row_dict)

    return index


def collect_external_lexicon_candidates_indexed(
    query_terms: Iterable[str],
    lexicon_index: dict,
    results_per_term: int = 20,
    min_score: float = 0.55,
) -> List[Dict]:
    """
    O(n_query_terms) indexed lookup used by input_and_context.py.
    """
    best_by_key: Dict = {}

    for term in query_terms:
        term_norm = normalize_text(term)
        if not term_norm:
            continue

        for row_dict in lexicon_index.get(term_norm, []):
            source_score = float(row_dict.get("score", 0.75))
            if source_score < min_score:
                continue

            lexicon_term = str(row_dict.get("term", "")).strip()
            candidate = {
                "source": "external_lexicon",
                "matched_query": term,
                "matched_lexicon_term": lexicon_term,
                "label_or_term": f"{row_dict.get('source_name', 'HurtLex_BN')}::{row_dict.get('category', 'hurtlex_bn')}",
                "sentence": str(row_dict.get(
                    "sentence",
                    "Bengali HurtLex lexical cue matched; inspect the meme context.",
                )),
                "source_score": source_score,
                "url": None,
                "qid": None,
                "raw_description": (
                    f"source_file={row_dict.get('source_file', '')}; "
                    f"citation={row_dict.get('citation_key', '')}; "
                    f"hurtlex_id={row_dict.get('hurtlex_id', '')}; "
                    f"level={row_dict.get('hurtlex_level', '')}; "
                    f"stereotype={row_dict.get('hurtlex_stereotype', '')}; "
                    f"category={row_dict.get('hurtlex_category_code', '')}"
                ),
            }

            key = (normalize_text(lexicon_term), candidate["label_or_term"])
            old = best_by_key.get(key)
            if old is None or candidate["source_score"] > old["source_score"]:
                best_by_key[key] = candidate

    deduped = list(best_by_key.values())
    deduped.sort(key=lambda x: x.get("source_score", 0.0), reverse=True)
    return deduped


# ============================================================
# 6. Prompt formatting
# ============================================================

def format_candidates_for_prompt(candidates: List[Dict], max_items: int = 5) -> str:
    if not candidates:
        return "No Bengali HurtLex terms matched the OCR text."

    lines: List[str] = []
    seen = set()

    for item in candidates:
        lexicon_term = str(item.get("matched_lexicon_term", item.get("matched_query", ""))).strip()
        matched_query = str(item.get("matched_query", "")).strip()
        label = str(item.get("label_or_term", "")).strip()
        sentence = str(item.get("sentence", "")).strip()
        score = item.get("source_score", "")

        if not lexicon_term or not label:
            continue

        key = (normalize_text(lexicon_term), label)
        if key in seen:
            continue
        seen.add(key)

        if matched_query and normalize_text(matched_query) != normalize_text(lexicon_term):
            matched_line = f'  Matched OCR span: "{matched_query}"\n'
        else:
            matched_line = ""

        lines.append(
            f'- Matched Bengali HurtLex term: "{lexicon_term}"\n'
            f"{matched_line}"
            f"  Bengali HurtLex category: {label}\n"
            f"  Cue meaning: {sentence}\n"
            f"  Match score: {score}"
        )

        if len(lines) >= max_items:
            break

    if not lines:
        return "No Bengali HurtLex terms matched the OCR text."

    return "\n".join(lines)


def retrieve_lexicon_prompt_context(
    meme_text: str,
    image_caption: str = "",
    lexicon_dir: Optional[Path] = None,
    max_items: int = 5,
    min_score: float = 0.55,
    results_per_term: int = 20,
    include_builtin: bool = False,
) -> Dict:
    """
    Retrieve Bengali HurtLex lexical cues for teacher prompts or student context.
    """
    full_query = f"{meme_text or ''} {image_caption or ''}".strip()
    seed_terms = simple_tokenize(full_query)
    query_terms = build_query_terms_for_lexicons(seed_terms, full_query, max_n=4)

    lexicon_df = load_external_lexicon_knowledge(
        lexicon_dir=lexicon_dir,
        include_builtin=False,
    )

    candidates = collect_external_lexicon_candidates(
        query_terms=query_terms,
        lexicon_df=lexicon_df,
        results_per_term=results_per_term,
        min_score=min_score,
    )

    return {
        "prompt_text": format_candidates_for_prompt(candidates, max_items=max_items),
        "candidates": candidates,
        "num_candidates": len(candidates),
        "query_terms": query_terms,
    }


# ============================================================
# 7. Standalone check
# ============================================================

if __name__ == "__main__":
    df = load_external_lexicon_knowledge(lexicon_dir=EXTERNAL_LEXICON_DIR, include_builtin=False)
    print(f"Loaded {len(df)} Bengali HurtLex rows from: {EXTERNAL_LEXICON_DIR}")

    if len(df) > 0:
        print(df[["source_name", "category"]].value_counts().head(20))
    else:
        print("No Bengali HurtLex rows loaded. Expected file: external_lexicons_bn_only/hurtlex_BN.tsv")
