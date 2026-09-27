"""
Build a unified binary hateful/not-hateful meme dataset.
But we later only used the FHM ONLY. The other datasets were only used for preliminary experiments and analysis.
Sources: 
-------
- FHM         : cs5242-hateful-memes/hateful-memes-data  (HuggingFace)
- HarMeme     : samuelmathew28/harmeme                   (Kaggle)
- MAMI        : SemEval-2022 Task 5                      (local dir — see MAMI_DIR)
- MultiOFF    : Multimodal Offensive Meme Classification (local dir — see MULTIOFF_DIR)
- Multi3Hate  : Multilingual Multicultural Hate Memes    (HuggingFace — see MULTI3HATE_HF_ID)

Output: /media/alexis/SharedData/meme_dataset/
  images/           all images, named {source}_{id}.{ext}
  train.jsonl
  val.jsonl
  test.jsonl

Each JSONL line:
  {"id": str, "image_path": str, "text": str, "label": int, "source": str}
  label: 0 = not hateful, 1 = hateful

Prerequisites
-------------
  pip install datasets kaggle pandas pillow openpyxl tqdm
  Kaggle API credentials in ~/.kaggle/kaggle.json  (https://www.kaggle.com/settings → API)

Before running
--------------
  MAMI     : request via https://github.com/MIND-Lab/MAMI, extract to MAMI_DIR
  MultiOFF : clone https://github.com/bharathichezhiyan/Multimodal-Meme-Classification-Identifying-Offensive-Content-in-Image-and-Text
             and set MULTIOFF_DIR to the repo root
  Multi3Hate: verify MULTI3HATE_HF_ID on HuggingFace before running
"""

import json
import os
import shutil
from pathlib import Path
from typing import cast

# Redirect all HuggingFace I/O away from the main disk before importing datasets
os.environ.setdefault("HF_HOME", "/media/alexis/SharedData/tmp/hf_cache")

# Compatibility shims for mismatched Pillow versions and dataset scripts
import PIL.Image
import PIL.ExifTags
if not hasattr(PIL.Image, "ExifTags"):
    PIL.Image.ExifTags = PIL.ExifTags
if not hasattr(PIL.ExifTags, "Base"):
    import enum
    PIL.ExifTags.Base = enum.IntEnum(
        "Base",
        {name: tag for tag, name in PIL.ExifTags.TAGS.items() if str(name).isidentifier()},
    )

from tqdm import tqdm

BASE         = Path("/media/alexis/SharedData")
RAW_DIR      = BASE / "meme_dataset_raw"        # original downloaded files
SEP_DIR      = BASE / "meme_dataset_separated"  # per-source images + jsonl
OUT_DIR      = BASE / "meme_dataset"            # final merged output
HF_CACHE     = BASE / "tmp" / "hf_cache"

# ── Local dataset directories ─────────────────────────────────────────────
MAMI_DIR     = RAW_DIR / "mami_raw"      # SemEval-2022 Task 5
MULTIOFF_DIR = RAW_DIR / "multioff_raw"  # MultiOFF GitHub repo

# HuggingFace dataset ID for Multi3Hate — verify on https://huggingface.co/datasets
MULTI3HATE_HF_ID = "MinhDucBui/Multi3Hate"

# ── Label maps ────────────────────────────────────────────────────────────

HARMEME_LABEL_MAP = {
    "very_harmful":      1,
    "very harmful":      1,
    "partially_harmful": 1,
    "partially harmful": 1,
    "somewhat_harmful":  1,
    "somewhat harmful":  1,
    "harmful":           1,
    "harmless":          0,
    "not_harmful":       0,
    "not harmful":       0,
}

MULTIOFF_LABEL_MAP = {
    "offensive":     1,
    "non-offensiv":  0,   # truncated spelling in original CSV
    "non-offensive": 0,
    "not-offensive": 0,
}


# ── I/O helpers ───────────────────────────────────────────────────────────

def save_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def merge(dst: dict, src: dict) -> None:
    for split, rows in src.items():
        dst[split].extend(rows)


def _find_col(columns: list[str], *keywords: str) -> str | None:
    for kw in keywords:
        for c in columns:
            if kw in c:
                return c
    return None


def _source_img_dir(source: str) -> Path:
    """Return (and create) SEP_DIR/<source>/images/."""
    d = SEP_DIR / source / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_annotation_file(path: Path):
    import pandas as pd
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    elif path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    else:
        if path.suffix == ".jsonl" or path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            return pd.DataFrame(rows) if rows else None
        return pd.read_csv(path)


# ── FHM ───────────────────────────────────────────────────────────────────

def load_fhm() -> dict[str, list]:
    from datasets import load_dataset

    HF_CACHE.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("cs5242-hateful-memes/hateful-memes-data", cache_dir=str(HF_CACHE))

    split_map = {
        "train":       "train",
        "dev_seen":    "val",
        "dev_unseen":  "val",
        "test_seen":   "test",
        "test_unseen": "test",
    }

    records: dict[str, list] = {"train": [], "val": [], "test": []}

    for hf_split, out_split in split_map.items():
        if hf_split not in ds:
            print(f"  [FHM] split '{hf_split}' not found, skipping")
            continue
        for _row in tqdm(ds[hf_split], desc=f"  FHM {hf_split:<12}"):
            row = cast(dict, _row)
            img_id = f"fhm_{row['id']}"
            img_path = _source_img_dir("fhm") / f"{img_id}.png"
            if not img_path.exists():
                row["image"].save(img_path)
            records[out_split].append({
                "id":         img_id,
                "image_path": f"images/{img_id}.png",
                "text":       row["text"],
                "label":      int(row["label"]),
                "source":     "fhm",
            })

    return records


# ── HarMeme ───────────────────────────────────────────────────────────────

def load_harmeme() -> dict[str, list]:
    import kaggle
    import pandas as pd

    kaggle_dir = RAW_DIR / "_kaggle_harmeme"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    ann_files_check = list(kaggle_dir.rglob("*.jsonl")) + list(kaggle_dir.rglob("*.csv"))
    if not ann_files_check:
        print("  Downloading HarMeme from Kaggle …")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "samuelmathew28/harmeme", path=str(kaggle_dir), unzip=True, quiet=False
        )
    else:
        print(f"  HarMeme already downloaded ({len(ann_files_check)} annotation files found)")

    records: dict[str, list] = {"train": [], "val": [], "test": []}

    # Index every image file by filename — works regardless of subdirectory structure
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    img_lookup: dict[str, Path] = {
        f.name: f
        for f in kaggle_dir.rglob("*")
        if f.suffix.lower() in IMG_EXTS
    }
    print(f"  HarMeme image index: {len(img_lookup)} files found")

    ann_files = (
        list(kaggle_dir.rglob("*.csv"))
        + list(kaggle_dir.rglob("*.jsonl"))
        + list(kaggle_dir.rglob("*.json"))
    )

    if not ann_files:
        raise FileNotFoundError(f"No annotation files found under {kaggle_dir}")

    for ann_file in ann_files:
        name = ann_file.stem.lower()
        if name.startswith("target_"):
            continue  # target-group sub-task files, not hate labels
        if "train" in name:
            split = "train"
        elif "dev" in name or "val" in name:
            split = "val"
        elif "test" in name:
            split = "test"
        else:
            split = "train"

        # Determine sub-dataset so IDs and source stay unique across Harm-C / Harm-P
        path_parts = [p.lower() for p in ann_file.parts]
        if any("harm-c" in p for p in path_parts):
            subset = "harm-c"
        elif any("harm-p" in p for p in path_parts):
            subset = "harm-p"
        else:
            subset = "harmeme"

        df = _read_annotation_file(ann_file)
        if df is None or df.empty:
            continue

        df.columns = [str(c).lower().strip() for c in df.columns]
        cols = list(df.columns)

        label_col = _find_col(cols, "labels", "label", "class", "harm")
        img_col   = _find_col(cols, "image", "img", "file", "path", "fname")
        text_col  = _find_col(cols, "text", "caption", "ocr", "sentence")

        if label_col is None:
            print(f"  [HarMeme] No label column in {ann_file.name} (cols: {cols}) — skipping")
            continue

        img_base = next(
            (d for d in [
                ann_file.parent / "images",
                ann_file.parent / "img",
                ann_file.parent / "data",
            ] if d.is_dir()),
            None,
        ) or next(
            # Fall back to any 'images' dir found anywhere under the download root
            (d for d in sorted(kaggle_dir.rglob("images")) if d.is_dir()),
            ann_file.parent,
        )

        # Debug: show first row to verify paths and label parsing
        first = df.iloc[0]
        raw0 = first[label_col]
        if isinstance(raw0, list):
            raw0 = raw0[0] if raw0 else ""
        elif isinstance(raw0, str) and raw0.startswith("["):
            import ast
            try:
                raw0 = ast.literal_eval(raw0)[0]
            except Exception:
                pass
        sample_img_val = str(first[img_col]) if img_col else "N/A"
        sample_img_path = img_base / sample_img_val if img_col else None
        print(f"  [HarMeme] {ann_file.name}:")
        print(f"    label_col={label_col!r}  resolved={str(raw0)!r}  in_map={str(raw0).strip().lower() in HARMEME_LABEL_MAP}")
        print(f"    img_base={img_base}  sample_img={sample_img_val!r}  exists={sample_img_path.exists() if sample_img_path else False}")

        skipped_label = skipped_img = 0
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"  HarMeme {ann_file.name:<20}"):
            import pandas as pd
            raw = row[label_col]
            if isinstance(raw, list):
                raw = raw[0] if raw else ""
            elif isinstance(raw, str) and raw.startswith("["):
                import ast
                try:
                    parsed = ast.literal_eval(raw)
                    raw = parsed[0] if parsed else ""
                except Exception:
                    pass
            raw_label = str(raw).strip().lower()
            label = HARMEME_LABEL_MAP.get(raw_label)
            if label is None:
                skipped_label += 1
                continue

            img_id  = f"{subset}_{ann_file.stem}_{i}"
            img_dst = None

            if img_col and pd.notna(row.get(img_col)):
                img_val = str(row[img_col])
                src = img_base / img_val
                if not src.exists():
                    src = img_lookup.get(Path(img_val).name)
                if src and src.exists():
                    ext = src.suffix or ".jpg"
                    dst = _source_img_dir(subset) / f"{img_id}{ext}"
                    if not dst.exists():
                        shutil.copy2(src, dst)
                    img_dst = f"images/{img_id}{ext}"

            if img_dst is None:
                skipped_img += 1
                continue

            text = (
                str(row[text_col]).strip()
                if text_col and pd.notna(row.get(text_col))
                else ""
            )

            records[split].append({
                "id":         img_id,
                "image_path": img_dst,
                "text":       text,
                "label":      label,
                "source":     subset,
            })

        if skipped_label or skipped_img:
            print(f"    Skipped {skipped_label} (bad label) + {skipped_img} (missing image)")

    return records


# ── MAMI ──────────────────────────────────────────────────────────────────
#   SemEval-2022 Task 5 — Misogynous Meme Identification
#   label=1 → misogynous (treated as hateful), label=0 → not hateful
#
#   Expected layout under MAMI_DIR:
#     TRAINING/   ← training images + Training_meme_dataset.csv (or .tsv)
#     test/       ← test images + Test_meme_dataset.csv (with labels)

def load_mami() -> dict[str, list]:
    import pandas as pd

    if not MAMI_DIR.exists():
        raise FileNotFoundError(
            f"MAMI directory not found: {MAMI_DIR}\n"
            "Place the MAMI dataset at that path."
        )

    records: dict[str, list] = {"train": [], "val": [], "test": []}

    # ── training & trial splits (labels inline in CSV) ────────────────────
    for csv_path, split in [
        (MAMI_DIR / "training" / "training.csv", "train"),
        (MAMI_DIR / "trial"    / "trial.csv",    "val"),
    ]:
        if not csv_path.exists():
            print(f"  [MAMI] {csv_path.name} not found, skipping")
            continue
        df = pd.read_csv(csv_path, sep="\t", encoding="utf-8-sig")
        df.columns = [str(c).strip() for c in df.columns]
        img_dir = csv_path.parent
        text_col = next((c for c in df.columns if "text" in c.lower()), None)
        skipped = 0
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"  MAMI {csv_path.name:<20}"):
            try:
                label = int(float(str(row["misogynous"]).strip()))
            except (ValueError, KeyError):
                skipped += 1
                continue
            if label not in (0, 1):
                skipped += 1
                continue
            fname = str(row["file_name"]).strip()
            src = img_dir / fname
            if not src.exists():
                skipped += 1
                continue
            img_id = f"mami_{split}_{i}"
            dst = _source_img_dir("mami") / f"{img_id}{src.suffix}"
            if not dst.exists():
                shutil.copy2(src, dst)
            text = str(row[text_col]).strip() if text_col and pd.notna(row.get(text_col)) else ""
            records[split].append({
                "id":         img_id,
                "image_path": f"images/{img_id}{src.suffix}",
                "text":       text,
                "label":      label,
                "source":     "mami",
            })
        if skipped:
            print(f"    Skipped {skipped} rows")

    # ── test split (labels in test_labels.txt, text in test/test.csv) ─────
    test_csv    = MAMI_DIR / "test" / "test.csv"
    test_labels = MAMI_DIR / "test_labels.txt"
    if test_csv.exists() and test_labels.exists():
        df_text = pd.read_csv(test_csv, sep="\t", encoding="utf-8-sig")
        df_text.columns = [str(c).strip() for c in df_text.columns]
        # test_labels.txt: no header, tab-sep — file_name | misogynous | shaming | ...
        df_lbl = pd.read_csv(test_labels, sep="\t", header=None,
                              names=["file_name", "misogynous", "shaming",
                                     "stereotype", "objectification", "violence"])
        df = df_text.merge(df_lbl, on="file_name", how="inner")
        img_dir = test_csv.parent
        text_col = next((c for c in df.columns if "text" in c.lower()), None)
        skipped = 0
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"  MAMI test.csv             "):
            try:
                label = int(float(str(row["misogynous"]).strip()))
            except (ValueError, KeyError):
                skipped += 1
                continue
            fname = str(row["file_name"]).strip()
            src = img_dir / fname
            if not src.exists():
                skipped += 1
                continue
            img_id = f"mami_test_{i}"
            dst = _source_img_dir("mami") / f"{img_id}{src.suffix}"
            if not dst.exists():
                shutil.copy2(src, dst)
            text = str(row[text_col]).strip() if text_col and pd.notna(row.get(text_col)) else ""
            records["test"].append({
                "id":         img_id,
                "image_path": f"images/{img_id}{src.suffix}",
                "text":       text,
                "label":      label,
                "source":     "mami",
            })
        if skipped:
            print(f"    Skipped {skipped} rows")
    else:
        print(f"  [MAMI] test files missing (need test/test.csv + test_labels.txt)")

    return records


# ── MultiOFF ──────────────────────────────────────────────────────────────
#   Multimodal Meme Classification — Offensive Content Detection
#   label=1 → offensive (treated as hateful), label=0 → not hateful
#
#   Expected layout under MULTIOFF_DIR:
#     Split Dataset/Training_meme_dataset.csv  (+ Validation / Testing)
#     Labelled Images/  ← all .png images
#
#   Download: https://github.com/bharathichezhiyan/Multimodal-Meme-Classification-Identifying-Offensive-Content-in-Image-and-Text

def load_multioff() -> dict[str, list]:
    import pandas as pd

    if not MULTIOFF_DIR.exists():
        raise FileNotFoundError(
            f"MultiOFF directory not found: {MULTIOFF_DIR}\n"
            "Clone the GitHub repo and set MULTIOFF_DIR."
        )

    records: dict[str, list] = {"train": [], "val": [], "test": []}

    csv_files = list(MULTIOFF_DIR.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {MULTIOFF_DIR}")

    for csv_file in csv_files:
        name = csv_file.stem.lower()
        if "train" in name:
            split = "train"
        elif "val" in name or "dev" in name:
            split = "val"
        elif "test" in name:
            split = "test"
        else:
            split = "train"

        df = pd.read_csv(csv_file)
        df.columns = [str(c).lower().strip() for c in df.columns]
        cols = list(df.columns)

        label_col = _find_col(cols, "label", "class", "offensive")
        img_col   = _find_col(cols, "image_name", "image", "img", "file", "path")
        text_col  = _find_col(cols, "sentence", "text", "caption", "ocr")

        if label_col is None:
            print(f"  [MultiOFF] No label column in {csv_file.name} (cols: {cols}) — skipping")
            continue

        img_base = next(
            (d for d in [
                MULTIOFF_DIR / "Labelled Images",
                csv_file.parent / "images",
                csv_file.parent / "img",
                MULTIOFF_DIR / "images",
                csv_file.parent,
            ] if d.is_dir()),
            csv_file.parent,
        )

        skipped = 0
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"  MultiOFF {csv_file.name:<20}"):
            raw_label = str(row[label_col]).strip().lower()
            label = MULTIOFF_LABEL_MAP.get(raw_label)
            if label is None:
                # prefix match for truncated values like "non-offensiv"
                for k, v in MULTIOFF_LABEL_MAP.items():
                    if raw_label.startswith(k) or k.startswith(raw_label):
                        label = v
                        break
            if label is None:
                skipped += 1
                continue

            img_id  = f"multioff_{split}_{i}"
            img_dst = None

            if img_col and pd.notna(row.get(img_col)):
                src = img_base / str(row[img_col])
                if src.exists():
                    ext = src.suffix or ".jpg"
                    dst = _source_img_dir("multioff") / f"{img_id}{ext}"
                    if not dst.exists():
                        shutil.copy2(src, dst)
                    img_dst = f"images/{img_id}{ext}"

            if img_dst is None:
                skipped += 1
                continue

            text = (
                str(row[text_col]).strip()
                if text_col and pd.notna(row.get(text_col))
                else ""
            )

            records[split].append({
                "id":         img_id,
                "image_path": img_dst,
                "text":       text,
                "label":      label,
                "source":     "multioff",
            })

        if skipped:
            print(f"    Skipped {skipped} rows (unknown label or missing image)")

    return records


# ── Multi3Hate ────────────────────────────────────────────────────────────
#   Multilingual Multicultural Hate Meme Dataset
#   label=1 → hateful, label=0 → not hateful
#
#   Loaded from HuggingFace: MULTI3HATE_HF_ID
#   Verify the dataset ID at https://huggingface.co/datasets before running.

def load_multi3hate() -> dict[str, list]:
    from datasets import load_dataset
    from PIL import Image as PILImage

    HF_CACHE.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(MULTI3HATE_HF_ID, cache_dir=str(HF_CACHE))

    split_map = {
        "train":      "train",
        "validation": "val",
        "val":        "val",
        "test":       "test",
    }

    records: dict[str, list] = {"train": [], "val": [], "test": []}

    for hf_split, out_split in split_map.items():
        if hf_split not in ds:
            continue

        for i, _row in enumerate(tqdm(ds[hf_split], desc=f"  Multi3Hate {hf_split:<12}")):
            row = cast(dict, _row)
            # Resolve label — column name may vary across versions
            label = None
            for col in ("label", "hateful", "hate", "class", "is_hateful"):
                if col in row:
                    val = row[col]
                    if isinstance(val, int):
                        label = val
                    elif isinstance(val, float):
                        label = int(val)
                    else:
                        s = str(val).strip().lower()
                        label = 1 if s in {"hateful", "hate", "1", "yes", "true"} else 0
                    break
            if label is None:
                continue

            img_id  = f"multi3hate_{hf_split}_{i}"
            img_dst = None

            for img_col in ("image", "img", "pixel_values"):
                if img_col not in row or row[img_col] is None:
                    continue
                img = row[img_col]
                if not isinstance(img, PILImage.Image):
                    try:
                        img = PILImage.fromarray(img)
                    except Exception:
                        break
                img_path = _source_img_dir("multi3hate") / f"{img_id}.png"
                if not img_path.exists():
                    img.save(img_path)
                img_dst = f"images/{img_id}.png"
                break

            if img_dst is None:
                continue

            text = ""
            for txt_col in ("text", "caption", "ocr", "sentence", "transcription"):
                if txt_col in row and row[txt_col]:
                    text = str(row[txt_col]).strip()
                    break

            records[out_split].append({
                "id":         img_id,
                "image_path": img_dst,
                "text":       text,
                "label":      label,
                "source":     "multi3hate",
            })

    return records


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SEP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Run loaders — each writes images + JSONL into meme_dataset_separated/ ──
    loaders = [
        ("FHM",        load_fhm),
        ("HarMeme",    load_harmeme),
        ("MAMI",       load_mami),
        ("MultiOFF",   load_multioff),
        ("Multi3Hate", load_multi3hate),
    ]

    for name, loader in loaders:
        print()
        print("=" * 50)
        print(f"Loading {name} …")
        print("=" * 50)
        try:
            source_records = loader()
        except Exception as exc:
            print(f"  [{name}] Failed: {exc}")
            print(f"  Continuing without {name}.")
            continue

        by_source: dict[str, dict[str, list]] = {}
        for split, rows in source_records.items():
            for row in rows:
                src = row["source"]
                by_source.setdefault(src, {"train": [], "val": [], "test": []})
                by_source[src][split].append(row)
        for src, splits in by_source.items():
            src_dir = SEP_DIR / src
            src_dir.mkdir(parents=True, exist_ok=True)
            for split, rows in splits.items():
                if rows:
                    save_jsonl(rows, src_dir / f"{split}.jsonl")

    # ── Merge from meme_dataset_separated/ (reads disk, not memory) ───────
    # This means re-runs accumulate previously processed sources automatically,
    # even if some loaders fail in the current run.
    print()
    print("=" * 50)
    print("Merging from meme_dataset_separated/ …")
    print("=" * 50)

    merged_img_dir = OUT_DIR / "images"
    merged_img_dir.mkdir(parents=True, exist_ok=True)

    all_records: dict[str, list] = {"train": [], "val": [], "test": []}

    for src_dir in sorted(SEP_DIR.iterdir()):
        if not src_dir.is_dir():
            continue
        src_img_dir = src_dir / "images"
        n_copied = 0
        if src_img_dir.is_dir():
            for img in src_img_dir.iterdir():
                if img.is_file():
                    dst = merged_img_dir / img.name
                    if not dst.exists():
                        shutil.copy2(img, dst)
                        n_copied += 1
        n_rows = 0
        for split in ("train", "val", "test"):
            jsonl = src_dir / f"{split}.jsonl"
            if jsonl.exists():
                with open(jsonl, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_records[split].append(json.loads(line))
                            n_rows += 1
        print(f"  {src_dir.name:<15}: {n_copied} images copied, {n_rows} rows loaded")

    for split, records in all_records.items():
        save_jsonl(records, OUT_DIR / f"{split}.jsonl")
        n_hat = sum(r["label"] == 1 for r in records)
        n_not = sum(r["label"] == 0 for r in records)
        print(f"  {split:<6}: {len(records):>5} samples  "
              f"(hateful={n_hat}, not_hateful={n_not})")

    print(f"\nSeparated : {SEP_DIR}")
    print(f"Merged    : {OUT_DIR}")


if __name__ == "__main__":
    main()
