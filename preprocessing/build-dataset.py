"""
build_dataset.py
================
PURPOSE: Merge all raw data sources into one unified CSV.
         NO cleaning. NO preprocessing. Raw text only.
         All cleaning happens in preprocessing.py.

Run once from project root:
    python preprocessing/build_dataset.py

Output:
    data/processed/unified_raw.csv

Schema of output:
    text          : raw original text (no cleaning at all)
    label         : positive / negative / neutral
    source        : which dataset this row came from
    label_source  : how the label was assigned (manual / weak_stars / weak_lexicon / weak_lexicon_synthetic)
"""

import re
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "preprocessing" / "datasets" / "raw"
PROCESSED = ROOT / "preprocessing" / "datasets" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LABEL NORMALIZATION
# Standardizes all raw label formats to: positive / negative / neutral
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {
    "1": "positive", "pos": "positive", "positive": "positive",
    "Positive": "positive", "2": "positive", "4": "positive",
    "5": "positive", "-1": "negative", "neg": "negative",
    "negative": "negative", "Negative": "negative", "0": "negative",
    "neutral": "neutral", "Neutral": "neutral", "mixed": "neutral",
    "objective":"neutral", "3": "neutral",
}

def normalize_label(raw) -> str | None:
    return LABEL_MAP.get(str(raw).strip(), None)

def star_to_label(stars) -> str | None:
    try:
        s = int(float(str(stars)))
        if s >= 4: return "positive"
        if s <= 2: return "negative"
        return "neutral"
    except (ValueError, TypeError):
        return None

# ─────────────────────────────────────────────────────────────────────────────
# WEAK LABELING — AraSenti lexicon
# Only used for datasets that have no inherent labels.
# ─────────────────────────────────────────────────────────────────────────────

ARASENTI_PATH = Path("datasets/lexicon/AraSentiLexiconV1.0")

# Gulf-specific overrides added on top of AraSenti
GULF_OVERRIDES = {
    "وايد زين": 5.0, "زين": 4.5, "وايد": 0.0, "خوش": 4.0, "يسلم": 4.0,
    "مو زين": -4.0, "زفت": -4.5, "خربان": -4.0, "فاشل": -5.0,
    "excellent": 5.0, "amazing": 5.0, "great": 4.0, "good": 3.0,
    "bad": -4.5, "terrible": -5.0, "awful": -5.0, "worst": -5.0,
    "horrible": -5.0, "slow": -3.0, "broken": -4.0, "scam": -5.0,
}

def _load_lexicon() -> dict:
    lex = {}
    if ARASENTI_PATH.exists():
        try:
            with open(ARASENTI_PATH, "rb") as f:
                text = f.read().decode("cp1256")
            for line in text.split("\n")[70:]:
                line = line.strip()
                if not line: continue
                parts = re.split(r"\s{2,}|\t", line)
                if len(parts) >= 2:
                    try:
                        lex[parts[0].strip()] = float(parts[-1].strip())
                    except ValueError:
                        pass
            log.info("AraSenti lexicon: %d entries loaded", len(lex))
        except Exception as e:
            log.warning("AraSenti load failed: %s", e)
    lex.update(GULF_OVERRIDES)
    return lex

LEXICON = _load_lexicon()

def lexicon_label(text: str) -> str:
    tokens = str(text).lower().split()
    scores = [LEXICON[re.sub(r"[^\u0600-\u06FFa-zA-Z]", "", t)]
              for t in tokens
              if re.sub(r"[^\u0600-\u06FFa-zA-Z]", "", t) in LEXICON]
    if not scores:
        return "neutral"
    mean = float(np.mean(scores))
    if mean > 0.5:  return "positive"
    if mean < -0.5: return "negative"
    return "neutral"

# ─────────────────────────────────────────────────────────────────────────────
# FILE FINDERS
# ─────────────────────────────────────────────────────────────────────────────

def _find(pattern: str) -> Path | None:
    matches = list(RAW.glob(pattern))
    return matches[0] if matches else None

def _detect_text_col(df: pd.DataFrame) -> str:
    for kw in ["text", "tweet", "content", "review", "comment", "post", "body"]:
        match = next((c for c in df.columns if kw in c.lower()), None)
        if match: return match
    return df.columns[0]

def _detect_label_col(df: pd.DataFrame) -> str | None:
    for kw in ["label", "sentiment", "class", "polarity"]:
        match = next((c for c in df.columns if kw in c.lower()), None)
        if match: return match
    return None

def _detect_star_col(df: pd.DataFrame) -> str | None:
    for kw in ["star", "rating", "score"]:
        match = next((c for c in df.columns if kw in c.lower()), None)
        if match: return match
    return None

# ─────────────────────────────────────────────────────────────────────────────
# LOADERS — each returns DataFrame with columns: text, label, source, label_source
# Text is RAW — no cleaning at all.
# ─────────────────────────────────────────────────────────────────────────────

def load_astd() -> pd.DataFrame:
    log.info("Loading ASTD Tweets.txt...")
    path = RAW / "Tweets.txt"
    if not path.exists():
        log.warning("NOT FOUND"); return pd.DataFrame()
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("\t", 1) if "\t" in line else line.split(" ", 1)
            if len(parts) == 2:
                rows.append({"text": parts[1].strip(), "raw_label": parts[0].strip()})
    df = pd.DataFrame(rows)
    df["label"] = df["raw_label"].apply(normalize_label)
    df["source"] = "astd_tweets"
    df["label_source"] = "manual"
    log.info("  %d rows", len(df))
    return df[["text", "label", "source", "label_source"]]


def load_arbml() -> pd.DataFrame:
    log.info("Loading arbml parquet...")
    path = _find("arbml*.parquet") or _find("Arabic_Sentiment*.parquet")
    if not path:
        log.warning("NOT FOUND"); return pd.DataFrame()
    df = pd.read_parquet(path)
    log.info("%d rows | columns: %s", len(df), list(df.columns))
    text_col  = _detect_text_col(df)
    label_col = _detect_label_col(df)
    if label_col:
        out = df[[text_col, label_col]].copy()
        out.columns = ["text", "raw_label"]
        out["label"] = out["raw_label"].apply(normalize_label)
        out["label_source"] = "manual"
    else:
        out = df[[text_col]].copy()
        out.columns = ["text"]
        out["label"] = out["text"].apply(lexicon_label)
        out["label_source"] = "weak_lexicon"
    out["source"] = "arbml_twitter"
    log.info("  %d rows loaded", len(out))
    return out[["text", "label", "source", "label_source"]]


def load_magedsaeed() -> pd.DataFrame:
    log.info("Loading MagedSaeed parquet...")
    path = _find("MagedSaeed*.parquet") or _find("arabic-english*.parquet")
    if not path:
        log.warning("NOT FOUND"); return pd.DataFrame()
    df = pd.read_parquet(path)
    log.info("  %d rows | columns: %s", len(df), list(df.columns))
    out = df[[df.columns[0]]].copy()
    out.columns = ["text"]
    out["label"] = out["text"].apply(lexicon_label)
    out["source"] = "magedsaeed_cs"
    out["label_source"] = "weak_lexicon"
    log.info("  %d rows loaded", len(out))
    return out[["text", "label", "source", "label_source"]]


def load_company_reviews() -> pd.DataFrame:
    log.info("Loading CompanyReviews.csv...")
    path = RAW / "CompanyReviews.csv"
    if not path.exists():
        log.warning("  NOT FOUND"); return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    log.info("  %d rows", len(df))
    out = df[["review_description", "rating"]].copy()
    out.columns = ["text", "raw_label"]
    out["label"] = out["raw_label"].apply(normalize_label)
    out["source"] = "company_reviews"
    out["label_source"] = "manual"
    return out[["text", "label", "source", "label_source"]]


def load_appstore() -> pd.DataFrame:
    log.info("Loading appstore.csv...")
    path = RAW / "appstore.csv"
    if not path.exists():
        log.warning("  NOT FOUND"); return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    log.info("  %d rows | columns: %s", len(df), list(df.columns))
    text_col = _detect_text_col(df)
    star_col = _detect_star_col(df)
    label_col = _detect_label_col(df)
    out = df[[text_col]].copy(); out.columns = ["text"]
    if label_col:
        out["label"] = df[label_col].apply(normalize_label)
        out["label_source"] = "weak_stars"
    elif star_col:
        out["label"] = df[star_col].apply(star_to_label)
        out["label_source"] = "weak_stars"
    else:
        out["label"] = out["text"].apply(lexicon_label)
        out["label_source"] = "weak_lexicon"
    out["source"] = "appstore_scrape"
    return out[["text", "label", "source", "label_source"]]


def load_reddit() -> pd.DataFrame:
    log.info("Loading reddit_gulf.csv...")
    path = RAW / "reddit_gulf.csv"
    if not path.exists():
        log.warning("NOT FOUND"); return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    log.info("%d rows | columns: %s", len(df), list(df.columns))
    text_col = _detect_text_col(df)
    out = df[[text_col]].copy(); out.columns = ["text"]
    out["label"] = out["text"].apply(lexicon_label)
    out["source"] = "reddit_gulf"
    out["label_source"] = "weak_lexicon"
    return out[["text", "label", "source", "label_source"]]


def load_youtube() -> pd.DataFrame:
    log.info("Loading youtube_gulf.csv...")
    path = RAW / "youtube_gulf.csv"
    if not path.exists():
        log.warning("NOT FOUND"); return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    log.info("  %d rows | columns: %s", len(df), list(df.columns))
    text_col = _detect_text_col(df)
    out = df[[text_col]].copy(); out.columns = ["text"]
    out["label"] = out["text"].apply(lexicon_label)
    out["source"] = "youtube_scrape"
    out["label_source"] = "weak_lexicon_NEEDS_MANUAL"
    return out[["text", "label", "source", "label_source"]]


def load_are_cstd(filename: str, source_tag: str) -> pd.DataFrame:
    log.info("Loading %s...", filename)
    path = RAW / filename
    if not path.exists():
        log.warning("NOT FOUND"); return pd.DataFrame()
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [l.strip() for l in f if l.strip()]
    log.info("  %d lines", len(lines))
    out = pd.DataFrame({"text": lines})
    out["label"] = out["text"].apply(lexicon_label)
    out["source"] = source_tag
    out["label_source"] = "weak_lexicon_synthetic"
    return out[["text", "label", "source", "label_source"]]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_unified_raw():
    log.info("=" * 60)
    log.info("  Unified Raw Dataset Build")
    log.info("  NO preprocessing — raw text only")
    log.info("=" * 60)

    frames = [
        load_astd(),
        load_arbml(),
        load_magedsaeed(),
        load_company_reviews(),
        load_appstore(),
        load_reddit(),
        load_youtube(),
        load_are_cstd("SA_TRAIN.txt",  "are_cstd_saudi_train"),
        load_are_cstd("SA_TEST.txt",   "are_cstd_saudi_test"),
        load_are_cstd("MSA_TRAIN.txt", "are_cstd_msa_train"),
        load_are_cstd("MSA_TEST.txt",  "are_cstd_msa_test"),
    ]
    frames = [f for f in frames if not f.empty]

    # Merge
    df = pd.concat(frames, ignore_index=True)
    log.info("Merged: %d rows", len(df))

    # Minimal validation — only drop completely empty rows and unrecognized labels
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 3]       # drop near-empty
    df = df[df["text"] != "nan"]
    df = df[df["label"].notna()]
    df = df[df["label"] != "None"]

    # Drop pure English (not relevant to Arabic-English CS task)
    has_arabic  = df["text"].str.contains(r"[\u0600-\u06FF]", regex=True, na=False)
    has_english = df["text"].str.contains(r"[a-zA-Z]{2,}", regex=True, na=False)
    df = df[has_arabic | has_english]

    # Deduplicate on raw text
    before = len(df)
    df.drop_duplicates(subset="text", inplace=True)
    log.info("Deduplication removed: %d rows", before - len(df))

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    log.info("\nSUMMARY")
    log.info("Total rows : %d", len(df))
    log.info("Label counts :\n%s", df["label"].value_counts().to_string())
    log.info("Source counts :\n%s", df["source"].value_counts().to_string())

    out_path = PROCESSED / "unified_raw.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("Saved: %s", out_path)
    return df


if __name__ == "__main__":
    build_unified_raw()