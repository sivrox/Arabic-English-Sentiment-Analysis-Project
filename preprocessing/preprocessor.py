"""
preprocessing.py
================
PURPOSE: Takes unified_raw.csv and produces model-ready PyTorch DataLoaders.

Everything happens in this one file:
  1.  Load unified_raw.csv
  2.  Clean text       — emoji handling, noise removal
  3.  Normalize Arabic — unicode normalization, Gulf dialect normalization
  4.  Class balancing  — upsample neutral, downsample positive
  5.  Split            — train / val / test / gold_test
  6.  Tokenize         — convert text to token IDs using HuggingFace tokenizer
  7.  Pad + truncate   — make all sequences exactly max_length tokens
  8.  Convert labels   — string labels to integers (negative=0, neutral=1, positive=2)
  9.  Build tensors    — wrap everything in PyTorch tensors
  10. Build DataLoaders — batch the tensors for training

Usage in training notebook:
    from preprocessing import build_dataloaders

    train_loader, val_loader, test_loader = build_dataloaders(
        data_path   = "data/processed/unified_raw.csv",
        tokenizer   = xlm_roberta_tokenizer,
        batch_size  = 16,
        max_length  = 128,
    )
"""

import re
import unicodedata
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LABEL ENCODING
# negative=0, neutral=1, positive=2
# CrossEntropyLoss expects integer indices starting from 0.
# Using -1/0/1 would crash PyTorch because -1 is an invalid class index.
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — EMOJI HANDLING
# Maps sentiment emojis to text tokens instead of removing them.
# Emojis carry strong sentiment signal in Gulf Arabic social media.
# Unknown emojis are removed because the tokenizer fragments them into
# meaningless unicode escape sequences.
# ─────────────────────────────────────────────────────────────────────────────

EMOJI_MAP = {
    # Positive
    "😊": " [POS_EMOJI] ", "😄": " [POS_EMOJI] ", "😁": " [POS_EMOJI] ",
    "🥰": " [POS_EMOJI] ", "😍": " [POS_EMOJI] ", "🤩": " [POS_EMOJI] ",
    "😎": " [POS_EMOJI] ", "🙂": " [POS_EMOJI] ", "👍": " [POS_EMOJI] ",
    "✅": " [POS_EMOJI] ", "💯": " [POS_EMOJI] ", "🔥": " [POS_EMOJI] ",
    "❤": " [POS_EMOJI] ",  "⭐": " [POS_EMOJI] ", "🌟": " [POS_EMOJI] ",
    "🎉": " [POS_EMOJI] ", "🙏": " [POS_EMOJI] ", "👏": " [POS_EMOJI] ",
    "💪": " [POS_EMOJI] ", "🏆": " [POS_EMOJI] ", "✨": " [POS_EMOJI] ",
    "😋": " [POS_EMOJI] ", "😘": " [POS_EMOJI] ",
    # Negative
    "😢": " [NEG_EMOJI] ", "😭": " [NEG_EMOJI] ", "😤": " [NEG_EMOJI] ",
    "😠": " [NEG_EMOJI] ", "😡": " [NEG_EMOJI] ", "🤬": " [NEG_EMOJI] ",
    "👎": " [NEG_EMOJI] ", "❌": " [NEG_EMOJI] ", "💔": " [NEG_EMOJI] ",
    "🤮": " [NEG_EMOJI] ", "🤢": " [NEG_EMOJI] ", "😖": " [NEG_EMOJI] ",
    "😩": " [NEG_EMOJI] ", "🤦": " [NEG_EMOJI] ", "💩": " [NEG_EMOJI] ",
    "😒": " [NEG_EMOJI] ", "🙁": " [NEG_EMOJI] ",
    # Neutral
    "😐": " [NEU_EMOJI] ", "😑": " [NEU_EMOJI] ", "🤔": " [NEU_EMOJI] ",
    "🙄": " [NEU_EMOJI] ", "😏": " [NEU_EMOJI] ",
}

def _handle_emojis(text: str) -> str:
    """
    Replace known sentiment emojis with text tokens.
    Remove all other emojis using Unicode category detection.
    Unicode category 'So' = Symbol, Other — covers most emojis.
    """
    # Replace known emojis (longest match first for multi-char sequences)
    for emoji_char, token in sorted(EMOJI_MAP.items(), key=lambda x: -len(x[0])):
        text = text.replace(emoji_char, token)
    # Remove remaining unknown emojis
    return "".join(c for c in text if unicodedata.category(c) != "So")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — NOISE REMOVAL
# Removes social media noise that carries no sentiment signal.
# ─────────────────────────────────────────────────────────────────────────────

def _remove_noise(text: str) -> str:
    """
    Remove URLs, @mentions, RT markers, HTML entities.
    Retain hashtag TEXT (e.g. #WasteOfMoney) but remove the # symbol.
    Normalize whitespace.
    """
    text = re.sub(r"http\S+|www\S+", "", text)       # URLs
    text = re.sub(r"@\w+", "", text)                  # @mentions
    text = re.sub(r"^RT\s*:?\s*", "", text)           # RT marker
    text = re.sub(r"&\w+;", "", text)                 # HTML entities
    text = re.sub(r"#(\w+)", r"\1", text)             # remove # but keep text
    text = re.sub(r"\s+", " ", text).strip()          # normalize spaces
    return text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — ARABIC UNICODE NORMALIZATION
# Arabic has many unicode variants for the same letter.
# Without this, 'أحمد' and 'احمد' are treated as completely different tokens
# by the tokenizer, even though they represent the same name.
# ─────────────────────────────────────────────────────────────────────────────

# Map of Arabic unicode variants → canonical form
HAMZA_MAP = {
    "\u0623": "\u0627",  # أ → ا
    "\u0625": "\u0627",  # إ → ا
    "\u0622": "\u0627",  # آ → ا
    "\u0671": "\u0627",  # ٱ → ا
}
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670]")

def _normalize_unicode(text: str) -> str:
    """
    Normalize Arabic unicode variants:
    - Hamza-bearing alef variants (أ إ آ ٱ) → plain alef (ا)
    - Teh marbuta (ة) → heh (ه)  [Gulf dialect often writes this way]
    - Alef maqsura (ى) → yeh (ي)
    - Remove tatweel elongation (ـ) [not linguistic content]
    - Remove diacritics/tashkeel [not used in informal social media]
    """
    result = []
    for char in text:
        if char in HAMZA_MAP:
            result.append(HAMZA_MAP[char])
        elif char == "\u0629":    # ة teh marbuta
            result.append("\u0647")  # ه heh
        elif char == "\u0649":    # ى alef maqsura
            result.append("\u064A")  # ي yeh
        elif char == "\u0640":    # ـ tatweel
            pass                  # skip
        else:
            result.append(char)
    text = "".join(result)
    text = ARABIC_DIACRITICS.sub("", text)  # remove diacritics
    return text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — GULF DIALECT NORMALIZATION
# Gulf Arabic has no standardized orthography.
# The same word appears in many spellings across different users.
# Without this, وايد / وايت / وَايد are treated as 3 different words.
# ─────────────────────────────────────────────────────────────────────────────

GULF_MAP = {
    # Laughter normalization
    "هههههههه": "هه", "ههههههه": "هه", "هههههه": "هه",
    "ههههه": "هه",    "هههه": "هه",    "ههه": "هه",
    "هاهاها": "هه",   "هاها": "هه",
    # وايد variants
    "وايت": "وايد",   "وَايد": "وايد",
    # يبغي variants
    "يبي": "يبغي",    "يبغى": "يبغي",
    # زين variants
    "زيين": "زين",    "زِين": "زين",
    # Common Gulf abbreviations
    "انشالله": "ان شاء الله",
    "ماشاءالله": "ما شاء الله",
    "ترا": "ترى",
    "عسب": "عشان",    "عسبان": "عشان",
    "تبي": "تبغي",    "نبي": "نبغي",
    "شكراً": "شكرا",
}

def _normalize_gulf(text: str) -> str:
    """Apply Gulf Arabic dialect spelling normalizations."""
    for pattern, replacement in GULF_MAP.items():
        text = text.replace(pattern, replacement)
    # Collapse any remaining هه repetitions (e.g. هههه after earlier replacements)
    text = re.sub(r"هه(ه)+", "هه", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# FULL CLEANING PIPELINE
# All four steps applied in order.
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Apply the full preprocessing pipeline to a single text string.
    Order matters:
      1. Emoji handling first (before noise removal removes characters)
      2. Noise removal
      3. Unicode normalization
      4. Gulf dialect normalization
    """
    text = str(text)
    text = _handle_emojis(text)
    text = _remove_noise(text)
    text = _normalize_unicode(text)
    text = _normalize_gulf(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — CLASSIFY TEXT TYPE
# Used for analysis and for filtering the gold test set.
# Not used as a model input feature.
# ─────────────────────────────────────────────────────────────────────────────

def classify_text_type(text: str) -> str:
    """
    Returns:
        'code_switched'  — contains both Arabic script and English words
        'pure_arabic'    — Arabic script only
        'pure_english'   — English only (will be filtered out)
        'other'          — neither
    """
    has_ar = bool(re.search(r"[\u0600-\u06FF]", text))
    has_en = bool(re.search(r"[a-zA-Z]{2,}", text))
    if has_ar and has_en: return "code_switched"
    if has_ar:            return "pure_arabic"
    if has_en:            return "pure_english"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — CLASS BALANCING
# Applied to training set only. Val and test sets are left at natural ratios
# so evaluation results are honest.
#
# Problem: positive=38.5%, neutral=36.2%, negative=25.3%
# Strategy:
#   - Neutral: upsample to match negative count (sample with replacement)
#   - Positive: downsample to 1.5x negative count
#   - Negative: keep as-is (reference class)
# ─────────────────────────────────────────────────────────────────────────────

def balance_training_set(train_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Balance the training set classes.
    Applied to training split only — never val or test.
    """
    pos = train_df[train_df["label"] == "positive"]
    neg = train_df[train_df["label"] == "negative"]
    neu = train_df[train_df["label"] == "neutral"]

    neg_count = len(neg)
    pos_target = min(int(neg_count * 1.5), len(pos))
    neu_target = neg_count

    log.info("Class balancing (training only):")
    log.info("  Before: pos=%d neu=%d neg=%d", len(pos), len(neu), neg_count)

    pos_balanced = pos.sample(n=pos_target, random_state=seed)
    neu_balanced = neu.sample(n=neu_target, replace=True, random_state=seed)

    balanced = pd.concat([pos_balanced, neg, neu_balanced], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)

    log.info("  After:  pos=%d neu=%d neg=%d",
             len(pos_balanced), len(neu_balanced), neg_count)
    return balanced


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def assign_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Assign train/val/test/gold_test splits.

    Rules:
    - YouTube scrape → gold_test (needs manual annotation)
    - Real code-switched samples → 20% reserved for test
    - Remaining → 80% train / 10% val / 10% test
    """
    df = df.copy()
    df["split"] = None

    # YouTube → gold test
    df.loc[df["source"] == "youtube_scrape", "split"] = "gold_test"

    # Real human CS samples → partial test reservation
    real_cs_sources = {"company_reviews", "appstore_scrape",
                       "magedsaeed_cs", "reddit_gulf"}
    real_cs_mask = (
        (df["text_type"] == "code_switched") &
        (df["source"].isin(real_cs_sources)) &
        (df["split"].isna())
    )
    real_cs_idx  = df[real_cs_mask].index.tolist()
    test_size    = max(1, int(len(real_cs_idx) * 0.20))
    df.loc[real_cs_idx[:test_size], "split"] = "test"

    # Remaining → 80/10/10
    remaining = df[df["split"].isna()].index.tolist()
    n = len(remaining)
    t1, t2 = int(n * 0.80), int(n * 0.90)
    df.loc[remaining[:t1], "split"] = "train"
    df.loc[remaining[t1:t2], "split"] = "val"
    df.loc[remaining[t2:], "split"] = "test"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — PYTORCH DATASET
# Holds the tokenized, padded tensors for one split.
#
# Why tensors?
#   PyTorch models operate exclusively on torch.Tensor objects.
#   They cannot accept pandas DataFrames, numpy arrays, or Python lists.
#   The GPU only understands tensors — converting to tensors is the step
#   that moves data from CPU memory into a format the GPU can process.
#
# Why padding and truncation?
#   Your reviews have different lengths (some are 5 tokens, some are 300).
#   A model processes batches of 16 samples simultaneously.
#   A batch must be a rectangular matrix — every row the same length.
#   Padding: add [PAD] tokens to short sequences → length = max_length
#   Truncation: cut long sequences at max_length = 128
# ─────────────────────────────────────────────────────────────────────────────

class ArabicSentimentDataset(Dataset):
    """
    PyTorch Dataset for Arabic-English sentiment classification.

    Holds three tensors:
        input_ids      : shape (n_samples, max_length) — token IDs
        attention_mask : shape (n_samples, max_length) — 1=real token, 0=padding
        labels         : shape (n_samples,)            — integer class 0/1/2
    """

    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer,
        max_length: int = 128,
    ):
        """
        Args:
            texts:      list of cleaned text strings
            labels:     list of integer labels (0=negative, 1=neutral, 2=positive)
            tokenizer:  HuggingFace tokenizer (XLM-RoBERTa or MARBERT)
            max_length: sequences are padded or truncated to this length
        """
        # Tokenize all texts at once (batch tokenization is faster than one-by-one)
        encoded = tokenizer(
            list(texts),
            padding="max_length",     # pad short sequences to max_length
            truncation=True,          # cut sequences longer than max_length
            max_length=max_length,
            return_tensors="pt",      # return as PyTorch tensors directly
        )
        self.input_ids = encoded["input_ids"]       # (n, max_length) int64 tensor
        self.attention_mask = encoded["attention_mask"]  # (n, max_length) int64 tensor
        self.labels = torch.tensor(list(labels), dtype=torch.long)  # (n,) int64

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns one sample as a dict of tensors.
        The DataLoader stacks these into batches automatically.
        """
        return {
            "input_ids" : self.input_ids[idx],       # (max_length,) tensor
            "attention_mask": self.attention_mask[idx],   # (max_length,) tensor
            "label" : self.labels[idx],           # scalar tensor
        }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# This is the only function your training notebook needs to call.
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    data_path:   str | Path,
    tokenizer,
    batch_size:  int  = 16,
    max_length:  int  = 128,
    seed:        int  = 42,
    save_cleaned_csv: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Full preprocessing pipeline — from raw CSV to DataLoaders.

    Steps performed inside this function:
        1. Load unified_raw.csv
        2. Clean all text (emoji → tokens, noise removal)
        3. Normalize Arabic (unicode + Gulf dialect)
        4. Classify text type (code_switched / pure_arabic)
        5. Drop pure-English rows
        6. Drop very short rows (< 5 chars after cleaning)
        7. Assign train/val/test splits
        8. Balance training set classes
        9. Tokenize with the provided tokenizer
        10. Pad + truncate to max_length
        11. Convert to PyTorch tensors
        12. Wrap in DataLoaders

    Args:
        data_path:        path to unified_raw.csv (output of build_dataset.py)
        tokenizer:        HuggingFace tokenizer — must match the model being trained
                          (use xlm-roberta tokenizer for XLM-RoBERTa,
                           marbert tokenizer for MARBERT)
        batch_size:       number of samples per training batch (default 16)
        max_length:       token sequence length after padding/truncation (default 128)
        seed:             random seed for reproducibility
        save_cleaned_csv: if True, saves cleaned text to data/processed/cleaned_dataset.csv

    Returns:
        (train_loader, val_loader, test_loader)
        Each is a PyTorch DataLoader yielding batches of:
            {"input_ids": tensor, "attention_mask": tensor, "label": tensor}

    Note:
        Call this function TWICE if training both XLM-RoBERTa and MARBERT:
        once with the XLM tokenizer and once with the MARBERT tokenizer.
        The cleaning/normalization steps are the same; only tokenization differs.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"Run preprocessing/build_dataset.py first."
        )

    log.info("=" * 55)
    log.info("Preprocessing Pipeline...")
    log.info("Tokenizer: %s", getattr(tokenizer, "name_or_path", "unknown"))
    log.info("=" * 55)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    log.info("Step 1: Loading %s...", data_path.name)
    df = pd.read_csv(data_path)
    log.info("  Loaded: %d rows", len(df))

    # ── 2-3. Clean and normalize ──────────────────────────────────────────────
    log.info("Step 2-3: Cleaning and normalizing text...")
    df["cleaned_text"] = df["text"].apply(clean_text)

    # ── 4-5. Filter by text type ──────────────────────────────────────────────
    log.info("Step 4-5: Classifying text type and filtering...")
    df["text_type"] = df["cleaned_text"].apply(classify_text_type)
    before = len(df)
    df = df[df["text_type"].isin(["code_switched", "pure_arabic"])]
    log.info("Dropped %d pure-English / other rows", before - len(df))

    # ── 6. Drop very short rows ───────────────────────────────────────────────
    before = len(df)
    df = df[df["cleaned_text"].str.len() >= 5]
    log.info("Dropped %d near-empty rows", before - len(df))

    log.info("Remaining: %d rows", len(df))

    # ── 7. Assign splits ──────────────────────────────────────────────────────
    log.info("Step 7: Assigning splits...")
    df = assign_splits(df, seed=seed)
    for sp, count in df["split"].value_counts().items():
        log.info("  %s: %d", sp, count)

    # Save cleaned CSV for reference (useful for error analysis later)
    if save_cleaned_csv:
        out_path = Path("data/processed/cleaned_dataset.csv")
        df[["cleaned_text", "label", "source", "label_source",
            "text_type", "split"]].to_csv(
            out_path, index=False, encoding="utf-8-sig"
        )
        log.info("Cleaned dataset saved: %s", out_path)

    # ── 8. Balance training set ───────────────────────────────────────────────
    log.info("Step 8: Balancing training set...")
    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()
    train_df = balance_training_set(train_df, seed=seed)

    # ── 9-12. Tokenize and build DataLoaders ──────────────────────────────────
    log.info("Steps 9-12: Tokenizing, padding, converting to tensors, building DataLoaders...")
    log.info("Max sequence length: %d tokens", max_length)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        log.info("  %s: %d samples | labels: %s",
                 split_name, len(split_df),
                 dict(split_df["label"].value_counts()))

    def _make_labels(split_df):
        labels = split_df["label"].map(LABEL_MAP)
        if labels.isna().any():
            log.warning("Some labels could not be mapped — dropping %d rows",
                        labels.isna().sum())
            split_df = split_df[labels.notna()].copy()
            labels   = labels.dropna()
        return split_df, labels.astype(int).tolist()

    train_df, train_labels = _make_labels(train_df)
    val_df,   val_labels   = _make_labels(val_df)
    test_df,  test_labels  = _make_labels(test_df)

    train_dataset = ArabicSentimentDataset(
        train_df["cleaned_text"].tolist(), train_labels, tokenizer, max_length
    )
    val_dataset   = ArabicSentimentDataset(
        val_df["cleaned_text"].tolist(),   val_labels,   tokenizer, max_length
    )
    test_dataset  = ArabicSentimentDataset(
        test_df["cleaned_text"].tolist(),  test_labels,  tokenizer, max_length
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=True
    )
    val_loader   = DataLoader(
        val_dataset,   batch_size=batch_size, shuffle=False, pin_memory=True
    )
    test_loader  = DataLoader(
        test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True
    )

    log.info("DataLoaders ready:")
    log.info("  train_loader: %d batches × batch_size %d", len(train_loader), batch_size)
    log.info("  val_loader  : %d batches", len(val_loader))
    log.info("  test_loader : %d batches", len(test_loader))
    log.info("=" * 55)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTION
# Returns the cleaned DataFrame (without DataLoaders) for inspection.
# ─────────────────────────────────────────────────────────────────────────────

def get_cleaned_dataframe(data_path: str | Path) -> pd.DataFrame:
    """
    Run cleaning and normalization only, return the cleaned DataFrame.
    Useful for inspection before training.
    Does not tokenize.
    """
    df = pd.read_csv(data_path)
    df["cleaned_text"] = df["text"].apply(clean_text)
    df["text_type"]    = df["cleaned_text"].apply(classify_text_type)
    df = df[df["text_type"].isin(["code_switched", "pure_arabic"])]
    df = df[df["cleaned_text"].str.len() >= 5]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE DEMO
# Run this file directly to verify the pipeline works
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # ── Text demo ────────────────────────────────────────────────────────────
    print("\nProcessed Data Sample:\n")
    test_cases = [
        "وايت كثييييير هههههههه الخدمة سيئة 😡",
        "أحسن تطبيق إستخدمته والله ❤ amazing app",
        "الـ delivery was really slow وايد متأخر",
        "https://example.com @user RT: check this out",
        "Service was good but التطبيق crashed twice",
    ]
    print(f"{'Original':<55}  {'Cleaned':<55}  Type")
    print("-" * 120)
    for text in test_cases:
        cleaned = clean_text(text)
        ttype   = classify_text_type(cleaned)
        print(f"{text[:55]:<55}  {cleaned[:55]:<55}  {ttype}")

    # ── Save cleaned dataset ─────────────────────────────────────────────────
    # Determine data path — check common locations
    ROOT = Path(__file__).resolve().parent.parent
    possible_paths = [
        ROOT / "preprocessing" / "datasets" / "processed" / "unified_raw.csv",
        Path("preprocessing/datasets/processed/unified_raw.csv"),
        Path("../preprocessing/datasets/processed/unified_raw.csv"),
    ]

    data_path = None
    for p in possible_paths:
        if p.exists():
            data_path = p
            break

    if data_path is None:
        print("\nFile not found.")
        sys.exit(1)

    print(f"\nCleaning dataset...")
    print(f"Input: {data_path}")

    df = get_cleaned_dataframe(data_path)

    # Assign splits so the output has a split column
    df = assign_splits(df, seed=42)

    # Save
    out_path = data_path.parent / "cleaned_dataset.csv"
    df[["cleaned_text", "label", "source", "label_source",
        "text_type", "split"]].to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )

    print(f"\nSaved: {out_path}")
    print(f"Total rows : {len(df):,}")
    print(f"\nLabel distribution:")
    print(df["label"].value_counts().to_string())
    print(f"\nText type distribution:")
    print(df["text_type"].value_counts().to_string())
    print(f"\nSplit distribution:")
    print(df["split"].value_counts().to_string())
    print(f"\nSample cleaned rows (code-switched):")
    cs = df[df["text_type"] == "code_switched"].sample(
        min(5, len(df[df["text_type"] == "code_switched"])), random_state=42
    )
    for _, row in cs.iterrows():
        print(f"  [{row['label']}] {row['cleaned_text'][:80]}")