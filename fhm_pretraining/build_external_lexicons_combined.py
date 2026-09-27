"""
build_external_lexicons_combined.py

Builds a single combined CSV from the separate uploaded lexicon files.
The combined CSV is optional: input_and_context.py can also load the separate
files directly through external_lexicon_knowledge.py.

Run:
    python build_external_lexicons_combined.py
"""

from pathlib import Path
from external_lexicon_knowledge import save_combined_lexicon_csv, load_external_lexicon_knowledge

ROOT = Path(__file__).resolve().parent
LEXICON_DIR = ROOT / "external_lexicons"
OUTPUT_PATH = LEXICON_DIR / "external_lexicons_combined.csv"

if __name__ == "__main__":
    df = load_external_lexicon_knowledge(LEXICON_DIR, include_builtin=True)
    print("Rows:", len(df))
    print(df["source_name"].value_counts().to_string())
    save_combined_lexicon_csv(OUTPUT_PATH, lexicon_dir=LEXICON_DIR)
    print("Saved:", OUTPUT_PATH)
