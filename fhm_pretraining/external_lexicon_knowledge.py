"""
external_lexicon_knowledge.py

Local lexical/social external knowledge retrieval for hateful meme experiments.

This module is intentionally separate from input_and_context.py so that the
original preprocessing pipeline stays readable and close to the original:
    image/text preprocessing + Wikidata/Wikipedia/GenericsKB retrieval

This module adds a fourth optional source:
    external_lexicon

It loads:
    1. Built-in social cue templates.
    2. The uploaded Plaza/HurtLex-derived English lexicon files:
       - xenophobia_lexicon_en.txt
       - immigrant_lexicon_en.txt
       - insults_lexicon_en.txt
       - misogyny_lexicon_en.txt
    3. MOL: Multilingual Offensive Lexicon annotated with contextual information.

Important modeling note:
    These lexicons should NOT be treated as labels. A matched term only produces
    a neutral context sentence such as "check whether this cue targets a social
    group". 

Citations of Lexicons:

Plaza/HurtLex-derived lexicons:
    Plaza-Del-Arco, Flor-Miriam, Molina-González, M. Dolores,
    Ureña-López, L. Alfonso, and Martín-Valdivia, M. Teresa. 2020.
    Detecting Misogyny and Xenophobia in Spanish Tweets Using Language Technologies.
    ACM Transactions on Internet Technology, 20(2), 1--19.

    Bassignana, Elisa, Basile, Valerio, and Patti, Viviana. 2018.
    HurtLex: A Multilingual Lexicon of Words to Hurt.
    Proceedings of the 5th Italian Conference on Computational Linguistics.

MOL:
    Vargas, Francielle, Carvalho, Isabelle, Pardo, Thiago A. S.,
    and Benevenuto, Fabrício. 2024.
    Context-aware and expert data resources for Brazilian Portuguese hate speech detection.
    Natural Language Processing, 31(2), 435--456.
    DOI: 10.1017/nlp.2024.18
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, List, Dict, Optional

import pandas as pd


# -----------------------------
# Metadata / citations
# -----------------------------

CITATIONS = {
    "plaza2020detecting": {
        "short": "Plaza-Del-Arco et al. (2020)",
        "bibtex": """@article{plaza2020detecting,
  title={Detecting Misogyny and Xenophobia in Spanish Tweets Using Language Technologies},
  author={Plaza-Del-Arco, Flor-Miriam and Molina-Gonz{\\'a}lez, M Dolores and Ure{\\~n}a-L{\\'o}pez, L Alfonso and Mart{\\'\\i}n-Valdivia, M Teresa},
  journal={ACM Transactions on Internet Technology (TOIT)},
  volume={20},
  number={2},
  pages={1--19},
  year={2020},
  publisher={ACM New York, NY, USA}
}""",
    },
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
    "vargas2024mol": {
        "short": "Vargas et al. (2024)",
        "bibtex": """@article{Vargas_Carvalho_Pardo_Benevenuto_2024,
  author={Vargas, Francielle and Carvalho, Isabelle and Pardo, Thiago A. S. and Benevenuto, Fabrício},
  title={Context-aware and expert data resources for Brazilian Portuguese hate speech detection},
  DOI={10.1017/nlp.2024.18},
  journal={Natural Language Processing},
  year={2024},
  pages={435--456},
  volume={31},
  number={2}
}""",
    },
}


# -----------------------------
# Built-in social cue templates
# -----------------------------

BUILTIN_SOCIAL_CUES = [
    {
        "source_name": "builtin_social_cues",
        "source_file": "builtin",
        "citation_key": "manual_task_templates",
        "term": "othering language",
        "aliases": "them;they;those people;these people;their kind;them young",
        "category": "othering",
        "sentence": "Othering language can separate an out-group from an in-group; check whether the meme uses this separation to target or stereotype a social group.",
        "score": 1.0,
    },
    {
        "source_name": "builtin_social_cues",
        "source_file": "builtin",
        "citation_key": "manual_task_templates",
        "term": "group generalization",
        "aliases": "all;all of them;always;never;every;entire group;whole group",
        "category": "stereotype",
        "sentence": "Broad generalizations about an entire group can signal stereotyping; check whether a negative trait is applied because of identity or group membership.",
        "score": 0.95,
    },
    {
        "source_name": "builtin_social_cues",
        "source_file": "builtin",
        "citation_key": "manual_task_templates",
        "term": "identity group",
        "aliases": "black;white;asian;jewish;muslim;christian;immigrant;refugee;women;men;gay;lesbian;lesbians;lgbt;lgbtq;queer;trans;disabled;disability;nationality;ethnicity;religion;gender;pride;pride month",
        "category": "target_group",
        "sentence": "References to identity categories can identify a target; check whether the meme attacks a group rather than discussing an individual action.",
        "score": 1.0,
    },
    {
        "source_name": "builtin_social_cues",
        "source_file": "builtin",
        "citation_key": "manual_task_templates",
        "term": "criminal framing",
        "aliases": "crime;criminal;criminals;thief;steal;rob;arrest;capturing;capture;prison;jail;offender;illegal",
        "category": "crime_association",
        "sentence": "Crime-related wording can become harmful when it falsely associates a social group with criminality; check whether the meme frames a group as naturally criminal or dangerous.",
        "score": 1.0,
    },
    {
        "source_name": "builtin_social_cues",
        "source_file": "builtin",
        "citation_key": "manual_task_templates",
        "term": "violence framing",
        "aliases": "violent;violence;attack;beat;beat up;beating;assault;hurt;harm;kill;killer;murder;terror;terrorist;threat;dangerous;weapon;war",
        "category": "violence_association",
        "sentence": "Violence-related wording can signal harmful framing when it associates a group with threat, aggression, or terrorism without evidence.",
        "score": 1.0,
    },
    {
        "source_name": "builtin_social_cues",
        "source_file": "builtin",
        "citation_key": "manual_task_templates",
        "term": "dehumanization",
        "aliases": "animal;ape;monkey;pig;dog;rat;snake;goat;goats;beast;creature;zoo;pest;parasite;dirty;filthy;disease;virus;plague",
        "category": "dehumanization",
        "sentence": "Animal, disease, contamination, or pest metaphors can dehumanize people; check whether the comparison is aimed at a protected or social group.",
        "score": 0.95,
    },
    {
        "source_name": "builtin_social_cues",
        "source_file": "builtin",
        "citation_key": "manual_task_templates",
        "term": "misogyny objectification",
        "aliases": "woman;women;girl;girls;wife;wives;female;females;kitchen;sandwich;handjob;handjobs;objectify;objectifying;sex object",
        "category": "misogyny_objectification",
        "sentence": "Gendered or sexualized framing can signal misogyny or objectification when it reduces women to domestic roles, bodies, sexuality, or inferiority.",
        "score": 0.95,
    },
    {
        "source_name": "builtin_social_cues",
        "source_file": "builtin",
        "citation_key": "manual_task_templates",
        "term": "sarcasm irony",
        "aliases": "yeah right;sure;totally;good guy;nice job;respect;thanks;peaceful;innocent",
        "category": "sarcasm_irony",
        "sentence": "Positive wording can be ironic when the image suggests the opposite; check whether text and image sentiment conflict in a way that attacks a target.",
        "score": 0.9,
    },
    {
        "source_name": "builtin_social_cues",
        "source_file": "builtin",
        "citation_key": "manual_task_templates",
        "term": "religious stereotype",
        "aliases": "72 virgins;virgins;goats;terrorist;jihad;muslim;islam;islamic",
        "category": "religious_stereotype",
        "sentence": "Religious references can become harmful when they mock, stereotype, or dehumanize people because of religion; check whether the meme targets a religious group.",
        "score": 0.95,
    },
]


PLAIN_LEXICON_CONFIGS = {
    "xenophobia_lexicon_en.txt": {
        "source_name": "xenophobia_lexicon",
        "category": "xenophobia_or_ethnic_slur_cue",
        "citation_key": "plaza2020detecting;bassignana2018hurtlex",
        "score": 0.90,
        "sentence": "A xenophobia-related or ethnic-offensive lexical cue matched; check whether the meme uses it to attack an immigrant, nationality, ethnic, racial, or religious group.",
    },
    "immigrant_lexicon_en.txt": {
        "source_name": "immigrant_lexicon",
        "category": "immigration_or_nationality_cue",
        "citation_key": "plaza2020detecting;bassignana2018hurtlex",
        "score": 0.65,
        "sentence": "An immigration, nationality, or foreignness-related term matched; check whether the meme frames immigrants or national groups as a threat or inferior.",
    },
    "insults_lexicon_en.txt": {
        "source_name": "insults_lexicon",
        "category": "general_insult_cue",
        "citation_key": "plaza2020detecting;bassignana2018hurtlex",
        "score": 0.70,
        "sentence": "A general insult or derogatory cue matched; check whether it is directed at a protected or social group rather than used as a generic insult.",
    },
    "misogyny_lexicon_en.txt": {
        "source_name": "misogyny_lexicon",
        "category": "misogyny_or_gendered_offense_cue",
        "citation_key": "plaza2020detecting;bassignana2018hurtlex",
        "score": 0.90,
        "sentence": "A misogyny-related or gendered offensive lexical cue matched; check whether the meme targets women or gender identity through objectification, inferiority, or sexualized insult.",
    },
}


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z][a-zA-Z\-']*\b", str(text).lower())


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    output = []
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
    phrases = []
    for n in range(1, max_n + 1):
        for i in range(0, max(0, len(tokens) - n + 1)):
            phrases.append(" ".join(tokens[i:i+n]))
    return unique_preserve_order(phrases)


def build_query_terms_for_lexicons(query_terms: Iterable[str], full_query: str, max_n: int = 4) -> List[str]:
    """Use both model query terms and raw n-grams so filtered words like 'them' are still checked."""
    return unique_preserve_order(list(query_terms) + make_ngrams_from_query(full_query, max_n=max_n))


def normalize_external_lexicon_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    defaults = {
        "term": "",
        "aliases": "",
        "category": "external_lexicon",
        "sentence": "External lexical cue matched; check the meme context before using it as evidence.",
        "score": 0.75,
        "source_name": "external_lexicon",
        "source_file": "unknown",
        "citation_key": "",
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


def _read_plain_terms(path: Path) -> List[str]:
    if not path.exists():
        return []
    terms = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            term = line.strip()
            if not term or term.startswith("#"):
                continue
            terms.append(term)
    return unique_preserve_order(terms)


def load_plain_lexicon(path: Path, cfg: dict) -> pd.DataFrame:
    rows = []
    for term in _read_plain_terms(path):
        rows.append({
            "term": term,
            "aliases": "",
            "category": cfg["category"],
            "sentence": cfg["sentence"],
            "score": cfg["score"],
            "source_name": cfg["source_name"],
            "source_file": path.name,
            "citation_key": cfg["citation_key"],
        })
    return pd.DataFrame(rows)


def load_mol_lexicon(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "en-american-english" not in df.columns:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        term = str(row.get("en-american-english", "")).strip()
        if not term or term.lower() in {"nan", "none"}:
            continue

        contextual = row.get("en-contextual-label", None)
        hate_label = str(row.get("en-hate-label", "0")).strip()
        if hate_label.lower() in {"nan", "none", ""}:
            hate_label = "0"

        # MOL label convention in README: 1 = context-independent offensive, 0 = context-dependent.
        try:
            contextual_val = int(float(contextual))
        except Exception:
            contextual_val = 0

        if hate_label != "0":
            category = "mol_target_" + re.sub(r"[^a-z0-9]+", "_", hate_label.lower()).strip("_")
            score = 0.90
            sentence = "MOL marks this lexical item as related to a hate/offense target category; check whether the meme targets that group in context."
        elif contextual_val == 1:
            category = "mol_context_independent_offensive"
            score = 0.80
            sentence = "MOL marks this as a mostly context-independent offensive lexical item; check whether it is used to attack a person or group in the meme."
        else:
            category = "mol_context_dependent_offensive"
            score = 0.65
            sentence = "MOL marks this as context-dependent offensive language; check image-text context before treating it as harmful."

        rows.append({
            "term": term,
            "aliases": "",
            "category": category,
            "sentence": sentence,
            "score": score,
            "source_name": "MOL",
            "source_file": path.name,
            "citation_key": "vargas2024mol",
        })
    return pd.DataFrame(rows)


def load_external_lexicon_knowledge(lexicon_dir: Optional[Path] = None, include_builtin: bool = True) -> pd.DataFrame:
    """
    Load all available external lexicon rows.

    Expected folder structure:
        external_lexicons/
          xenophobia_lexicon_en.txt
          immigrant_lexicon_en.txt
          insults_lexicon_en.txt
          misogyny_lexicon_en.txt
          mol.csv
    """
    frames = []
    if include_builtin:
        frames.append(pd.DataFrame(BUILTIN_SOCIAL_CUES))

    if lexicon_dir is not None:
        lexicon_dir = Path(lexicon_dir)

        # Optional pre-built combined file. This is useful on the cluster if you
        # prefer to ship one CSV, but separate source files are still supported.
        combined_path = lexicon_dir / "external_lexicons_combined.csv"
        if combined_path.exists():
            try:
                frames.append(pd.read_csv(combined_path))
            except Exception as error:
                print(f"[Warning] Could not load {combined_path}: {error}")

        for filename, cfg in PLAIN_LEXICON_CONFIGS.items():
            path = lexicon_dir / filename
            if path.exists():
                frames.append(load_plain_lexicon(path, cfg))
        mol_path = lexicon_dir / "mol.csv"
        if mol_path.exists():
            frames.append(load_mol_lexicon(mol_path))

    if not frames:
        return normalize_external_lexicon_dataframe(pd.DataFrame())

    combined = pd.concat(frames, ignore_index=True)
    return normalize_external_lexicon_dataframe(combined)


def phrase_match_score(lexicon_phrase: str, query_term: str) -> float:
    phrase = normalize_text(lexicon_phrase)
    term = normalize_text(query_term)
    if not phrase or not term:
        return 0.0

    if phrase == term:
        return 1.0

    # Exact phrase containment for multi-word expressions.
    if " " in phrase:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, term):
            return 0.95
        return 0.0

    # For single words, require exact token match, not substring match.
    term_tokens = set(simple_tokenize(term))
    if phrase in term_tokens:
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

    candidates = []
    for _, row in lexicon_df.iterrows():
        phrases = [row["term"]] + split_aliases(row.get("aliases", ""))
        match = max(phrase_match_score(p, term) for p in phrases)
        if match < min_score:
            continue

        source_score = float(row.get("score", 0.75)) * float(match)
        candidates.append({
            "source": "external_lexicon",
            "matched_query": term,
            "label_or_term": f"{row.get('source_name', 'external')}::{row.get('category', 'external_lexicon')}",
            "sentence": row.get("sentence", "External lexical cue matched; check the meme context."),
            "source_score": source_score,
            "url": None,
            "qid": None,
            "raw_description": f"source_file={row.get('source_file', '')}; citation={row.get('citation_key', '')}",
        })

    candidates.sort(key=lambda x: x["source_score"], reverse=True)
    return candidates[:limit]


def collect_external_lexicon_candidates(
    query_terms: Iterable[str],
    lexicon_df: pd.DataFrame,
    results_per_term: int = 20,
    min_score: float = 0.55,
) -> List[Dict]:
    all_candidates = []
    for term in query_terms:
        all_candidates.extend(
            search_external_lexicons(
                term=term,
                lexicon_df=lexicon_df,
                limit=results_per_term,
                min_score=min_score,
            )
        )

    # Avoid duplicates in final top-k. If two sources give the same neutral
    # context sentence, keep the higher-scoring one.
    best_by_key = {}
    for item in all_candidates:
        key = (normalize_text(item.get("sentence", "")), item.get("label_or_term", ""))
        old = best_by_key.get(key)
        if old is None or item.get("source_score", 0.0) > old.get("source_score", 0.0):
            best_by_key[key] = item

    deduped = list(best_by_key.values())
    deduped.sort(key=lambda x: x.get("source_score", 0.0), reverse=True)
    return deduped


def build_lexicon_index(lexicon_df: pd.DataFrame) -> dict:
    """
    Build {phrase_norm: [row_dicts]} for O(1) term lookup.

    All lexicon phrases (terms + aliases) are indexed by their normalized form.
    Since build_query_terms_for_lexicons generates all n-grams up to n=4 from
    the full query, every phrase of length ≤ 4 words will appear as a direct
    query term — making a dict lookup sufficient to replace the O(n_rows) scan.
    """
    index: dict = {}
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
    O(n_query_terms) replacement for collect_external_lexicon_candidates.

    Because build_query_terms_for_lexicons already emits every 1–4-gram of the
    full query, every lexicon phrase of ≤ 4 words that appears in the text will
    appear as its own query term. A direct index lookup is therefore equivalent
    to the O(n_rows × n_aliases) scan in search_external_lexicons.
    """
    best_by_key: dict = {}

    for term in query_terms:
        term_norm = normalize_text(term)
        if not term_norm:
            continue

        for row_dict in lexicon_index.get(term_norm, []):
            # Every index lookup is an exact normalized match between query term
            # and the indexed phrase (term or alias), so match_score is always 1.0.
            source_score = float(row_dict.get("score", 0.75))
            if source_score < min_score:
                continue

            candidate = {
                "source": "external_lexicon",
                "matched_query": term,
                "label_or_term": (
                    f"{row_dict.get('source_name', 'external')}"
                    f"::{row_dict.get('category', 'external_lexicon')}"
                ),
                "sentence": str(row_dict.get(
                    "sentence",
                    "External lexical cue matched; check the meme context."
                )),
                "source_score": source_score,
                "url": None,
                "qid": None,
                "raw_description": (
                    f"source_file={row_dict.get('source_file', '')}; "
                    f"citation={row_dict.get('citation_key', '')}"
                ),
            }
            key = (normalize_text(candidate["sentence"]), candidate["label_or_term"])
            old = best_by_key.get(key)
            if old is None or candidate["source_score"] > old["source_score"]:
                best_by_key[key] = candidate

    deduped = list(best_by_key.values())
    deduped.sort(key=lambda x: x.get("source_score", 0.0), reverse=True)
    return deduped


def save_combined_lexicon_csv(output_path: Path, lexicon_dir: Optional[Path] = None) -> Path:
    df = load_external_lexicon_knowledge(lexicon_dir=lexicon_dir, include_builtin=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    lexicon_dir = here / "external_lexicons"
    df = load_external_lexicon_knowledge(lexicon_dir=lexicon_dir, include_builtin=True)
    print(f"Loaded {len(df)} external lexicon rows from: {lexicon_dir}")
    print(df[["source_name", "category"]].value_counts().head(20))
    out = save_combined_lexicon_csv(lexicon_dir / "external_lexicons_combined.csv", lexicon_dir=lexicon_dir)
    print(f"Saved combined CSV to: {out}")


# -----------------------------
# Prompt formatting helper
# -----------------------------

def format_candidates_for_prompt(candidates: List[Dict], max_items: int = 5) -> str:
    """
    Convert retrieved lexicon candidates into neutral prompt text for the teacher.

    We intentionally include only neutral context sentences, not a claim that the
    meme is hateful. This prevents the lexicon from becoming a hard label.
    """
    if not candidates:
        return "No relevant external lexicon cues were retrieved."

    lines = []
    seen_sentences = set()
    for item in candidates:
        sentence = str(item.get("sentence", "")).strip()
        if not sentence:
            continue
        key = normalize_text(sentence)
        if key in seen_sentences:
            continue
        seen_sentences.add(key)
        lines.append(f"- {sentence}")
        if len(lines) >= max_items:
            break

    if not lines:
        return "No relevant external lexicon cues were retrieved."
    return "\n".join(lines)


def retrieve_lexicon_prompt_context(
    meme_text: str,
    image_caption: str = "",
    lexicon_dir: Optional[Path] = None,
    max_items: int = 5,
    min_score: float = 0.55,
    results_per_term: int = 20,
    include_builtin: bool = True,
) -> Dict:
    """
    Retrieve neutral external-lexicon cues for a teacher prompt.

    This is designed for generate_teacher_outputs.py. It does not load BLIP,
    BLIP-2, or GenericsKB. It only uses the meme OCR text plus an optional
    image caption if available.
    """
    full_query = f"{meme_text or ''} {image_caption or ''}".strip()
    seed_terms = simple_tokenize(full_query)
    query_terms = build_query_terms_for_lexicons(seed_terms, full_query, max_n=4)

    lexicon_df = load_external_lexicon_knowledge(
        lexicon_dir=lexicon_dir,
        include_builtin=include_builtin,
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
