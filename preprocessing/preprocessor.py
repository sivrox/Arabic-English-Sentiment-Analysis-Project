#PREPROCESSING PIPELINE (WITH DATALOADER CLASSES FOR PYTORCH AND HUGGINGFACE)

import re
import unicodedata
from pathlib import Path

import emoji
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

#Label encoding
label_map = {"negative": 0, "neutral": 1, "positive": 2}
id_to_label = {0: "negative", 1: "neutral", 2: "positive"}

#Emoji handling
#Known sentiment emojis are replaced with [POS_EMOJI], [NEG_EMOJI], [NEU_EMOJI] tokens
sentiment_emoji_map = {
    "😊": "[POS_EMOJI]", "😄": "[POS_EMOJI]", "😁": "[POS_EMOJI]",
    "🥰": "[POS_EMOJI]", "😍": "[POS_EMOJI]", "🤩": "[POS_EMOJI]",
    "😎": "[POS_EMOJI]", "🙂": "[POS_EMOJI]", "👍": "[POS_EMOJI]",
    "✅": "[POS_EMOJI]", "💯": "[POS_EMOJI]", "🔥": "[POS_EMOJI]",
    "❤":  "[POS_EMOJI]", "⭐": "[POS_EMOJI]", "🌟": "[POS_EMOJI]",
    "🎉": "[POS_EMOJI]", "🙏": "[POS_EMOJI]", "👏": "[POS_EMOJI]",
    "💪": "[POS_EMOJI]", "🏆": "[POS_EMOJI]", "✨": "[POS_EMOJI]",
    "😋": "[POS_EMOJI]", "😘": "[POS_EMOJI]",
    "😢": "[NEG_EMOJI]", "😭": "[NEG_EMOJI]", "😤": "[NEG_EMOJI]",
    "😠": "[NEG_EMOJI]", "😡": "[NEG_EMOJI]", "🤬": "[NEG_EMOJI]",
    "👎": "[NEG_EMOJI]", "❌": "[NEG_EMOJI]", "💔": "[NEG_EMOJI]",
    "🤮": "[NEG_EMOJI]", "🤢": "[NEG_EMOJI]", "😖": "[NEG_EMOJI]",
    "😩": "[NEG_EMOJI]", "🤦": "[NEG_EMOJI]", "💩": "[NEG_EMOJI]",
    "😒": "[NEG_EMOJI]", "🙁": "[NEG_EMOJI]",
    "😐": "[NEU_EMOJI]", "😑": "[NEU_EMOJI]", "🤔": "[NEU_EMOJI]",
    "🙄": "[NEU_EMOJI]", "😏": "[NEU_EMOJI]",
}

#Replace known sentiment emojis with tokens and remove all remaining emojis using the emoji library

def _handle_emojis(text):
    for char, token in sorted(sentiment_emoji_map.items(), key=lambda x: -len(x[0])):
        text = text.replace(char, f" {token} ")
    return emoji.replace_emoji(text, replace="")


#Noise removal
def _remove_noise(text):
    text = re.sub(r"http\S+|www\S+", "", text) #manage urls
    text = re.sub(r"@\w+", "", text) #@mentions
    text = re.sub(r"^RT\s*:?\s*", "", text) #reddit rt prefix
    text = re.sub(r"&\w+;", "", text) #html entities
    text = re.sub(r"#(\w+)", r"\1", text) #remove '#'
    return re.sub(r"\s+", " ", text).strip()


#Arabic unicode normalization
#Same Arabic lettercan  exists under multiple unicode codepoints and without normalization the same word would appear as different tokens to the model
hamza_map = {
    "\u0623": "\u0627",  #أ → ا
    "\u0625": "\u0627",  #إ → ا
    "\u0622": "\u0627",  #آ → ا
    "\u0671": "\u0627"}  #ٱ → ا

diacritics_re = re.compile(r"[\u064B-\u0652\u0670]")

def _normalize_arabic(text):
    result = []
    for char in text:
        if char in hamza_map:
            result.append(hamza_map[char])
        elif char == "\u0629": # ة → ه (Gulf dialect convention)
            result.append("\u0647")
        elif char == "\u0649": # ى → ي
            result.append("\u064A")
        elif char == "\u0640": #tatweel elongation - no linguistic content
            pass
        else:
            result.append(char)
    return diacritics_re.sub("", "".join(result))


#Gulf Dialect Normalization
gulf_map = {
    "هههههههه": "هه", "ههههههه": "هه", "هههههه": "هه",
    "ههههه": "هه", "هههه": "هه", "ههه": "هه",
    "هاهاها": "هه", "هاها": "هه",
    "وايت": "وايد", "وَايد": "وايد",
    "يبي": "يبغي", "يبغى": "يبغي",
    "زيين": "زين", "زِين": "زين",
    "انشالله": "ان شاء الله",
    "ماشاءالله": "ما شاء الله",
    "ترا": "ترى",
    "عسب": "عشان", "عسبان": "عشان",
    "تبي": "تبغي", "نبي": "نبغي",
    "شكراً": "شكرا",}

def _normalize_gulf(text):
    for pattern, replacement in gulf_map.items():
        text = text.replace(pattern, replacement)
    return re.sub(r"هه(ه)+", "هه", text)  #collapse leftover repeated هه


#Cleaning Pipeline Function
def clean_text(text):
    text = str(text)
    text = _handle_emojis(text)
    text = _remove_noise(text)
    text = _normalize_arabic(text)
    text = _normalize_gulf(text)
    return re.sub(r"\s+", " ", text).strip()


def classify_text_type(text):
    has_ar = bool(re.search(r"[\u0600-\u06FF]", text))     #Used for filtering and analysis (not used for training))
    has_en = bool(re.search(r"[a-zA-Z]{2,}", text))
    if has_ar and has_en: return "code_switched"
    if has_ar: return "pure_arabic"
    if has_en: return "pure_english"
    return "other"


#Dataset Split Assignment into Train, test and val

def assign_splits(df, seed=42):
    df = df.copy()
    df["split"] = None

    df.loc[df["source"] == "youtube_scrape", "split"] = "gold_test"     #YouTube comments will be used for manual annotation because of weak labels

    real_cs_sources = {"company_reviews", "appstore_scrape", "magedsaeed_cs", "reddit_gulf"}     # Reserving 20% of real code-switched samples specifically for test
    real_cs_mask = (
        (df["text_type"] == "code_switched") &
        (df["source"].isin(real_cs_sources)) &
        (df["split"].isna())
    )
    real_cs_idx = df[real_cs_mask].index.tolist()
    test_size = max(1, int(len(real_cs_idx) * 0.20))
    df.loc[real_cs_idx[:test_size], "split"] = "test"

    #Remaining data points
    remaining = df[df["split"].isna()].index.tolist()
    n = len(remaining)
    t1 = int(n * 0.80)
    t2 = int(n * 0.90)
    df.loc[remaining[:t1], "split"] = "train"
    df.loc[remaining[t1:t2], "split"] = "val"
    df.loc[remaining[t2:], "split"] = "test"

    return df


#Training Set Class Balancing

def balance_training_set(train_df, seed=42):
    pos = train_df[train_df["label"] == "positive"]
    neg = train_df[train_df["label"] == "negative"]
    neu = train_df[train_df["label"] == "neutral"]

    neg_count  = len(neg)
    pos_target = min(int(neg_count * 1.5), len(pos))
    neu_target = neg_count

    pos_bal = pos.sample(n=pos_target, random_state=seed)
    neu_bal = neu.sample(n=neu_target, replace=True, random_state=seed)

    balanced = pd.concat([pos_bal, neg, neu_bal], ignore_index=True)
    return balanced.sample(frac=1, random_state=seed).reset_index(drop=True)


#PYTORCH DATALOADER CLASS
#Tokenizes all texts once at initialization and stores them as tensors

class ArabicSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        encoded = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }


#Data Loader function to convert the dataset into pytorch dataloaders
def build_dataloaders(data_path, tokenizer, batch_size=16, max_length=128,
                      seed=42, save_cleaned_csv=True):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Run build_dataset.py first.")

    print(f"\nPreprocessing pipeline...")
    print(f"Input: {data_path}")
    print(f"Tokenizer: {getattr(tokenizer, 'name_or_path', 'unknown')}")

    #Load unified dataset
    df = pd.read_csv(data_path)
    print(f"Loaded: {len(df):,} rows")

    #Clean and normalize all text
    print("\nCleaning and normalizing text...")
    df["cleaned_text"] = df["text"].apply(clean_text)

    #Filter to Arabic-containing rows only
    df["text_type"] = df["cleaned_text"].apply(classify_text_type)
    before = len(df)
    df = df[df["text_type"].isin(["code_switched", "pure_arabic"])]
    df = df[df["cleaned_text"].str.len() >= 5]
    print(f"After filtering: {len(df):,} rows (removed {before - len(df):,})")

    #Assign splits
    df = assign_splits(df, seed=seed)
    print("\nSplit distribution:")
    for sp, count in df["split"].value_counts().items():
        print(f"  {sp}: {count:,}")

    #Save cleaned dataset
    if save_cleaned_csv:
        out_path = data_path.parent / "cleaned_dataset.csv"
        df[["cleaned_text", "label", "source", "label_source", "text_type", "split"]].to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Cleaned dataset saved: {out_path}")

    #Split and balance
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_df = balance_training_set(train_df, seed=seed)
    print(f"Training set after balancing: {len(train_df):,}")
    print(f"Label counts: {dict(train_df['label'].value_counts())}")

    #Map string labels to integers
    def _map_labels(split_df):
        labels = split_df["label"].map(label_map)
        valid  = labels.notna()
        if not valid.all():
            print(f"Warning: dropped {(~valid).sum()} rows with unrecognized labels")
        return split_df[valid], labels[valid].astype(int).tolist()

    train_df, train_labels = _map_labels(train_df)
    val_df, val_labels = _map_labels(val_df)
    test_df, test_labels = _map_labels(test_df)

    #Tokenize and build DataLoaders
    print("\nTokenizing...")
    train_ds = ArabicSentimentDataset(train_df["cleaned_text"].tolist(), train_labels, tokenizer, max_length)
    val_ds = ArabicSentimentDataset(val_df["cleaned_text"].tolist(), val_labels, tokenizer, max_length)
    test_ds = ArabicSentimentDataset(test_df["cleaned_text"].tolist(), test_labels, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    print(f"DataLoaders ready - Train: {len(train_loader):,} | Val: {len(val_loader):,} | Test: {len(test_loader):,} batches")
    return train_loader, val_loader, test_loader


#HUGGINGFACE DATA LOADER FUNCTION
#Same pipeline as build_dataloaders() but returns HuggingFace Dataset objects from its built-in dataset class

def build_hf_datasets(data_path, tokenizer, max_length=128,
                      seed=42, save_cleaned_csv=False):
    try:
        from datasets import Dataset as HFDataset
    except ImportError:
        raise ImportError("Install the datasets library: pip install datasets")

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}. Run build_dataset.py first.")

    print(f"\nStarting preprocessing pipeline (HuggingFace Trainer format)...")
    print(f"Tokenizer: {getattr(tokenizer, 'name_or_path', 'unknown')}")

    df = pd.read_csv(data_path)
    print(f"Loaded: {len(df):,} rows")

    #Cleaning and filtering
    print("\nCleaning text...")
    df["cleaned_text"] = df["text"].apply(clean_text)
    df["text_type"] = df["cleaned_text"].apply(classify_text_type)

    before = len(df)
    df = df[df["text_type"].isin(["code_switched", "pure_arabic"])]
    df = df[df["cleaned_text"].str.len() >= 5]
    print(f"Rows after filtering: {len(df):,}  (removed {before - len(df):,})")

    df = assign_splits(df, seed=seed)

    if save_cleaned_csv:
        out_path = data_path.parent / "cleaned_dataset.csv"
        df[["cleaned_text", "label", "source", "label_source", "text_type", "split"]].to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nCleaned dataset saved: {out_path}")

    train_df = balance_training_set(df[df["split"] == "train"].copy(), seed=seed)
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"Training set balanced: {len(train_df):,} samples")

    #Map labels to integers
    for split_df in [train_df, val_df, test_df]:
        split_df["labels"] = split_df["label"].map(label_map).astype(int)

    #Convert to HuggingFace Dataset and tokenize via .map()
    def _to_hf(split_df):
        hf_ds = HFDataset.from_pandas(split_df[["cleaned_text", "labels"]].reset_index(drop=True))
        def tokenize_batch(batch):
            return tokenizer(batch["cleaned_text"], padding="max_length", truncation=True, max_length=max_length)
        hf_ds = hf_ds.map(tokenize_batch, batched=True, batch_size=256)
        hf_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return hf_ds

    print("\nTokenizing datasets...")
    train_ds = _to_hf(train_df)
    val_ds = _to_hf(val_df)
    test_ds = _to_hf(test_df)

    print(f"\nDone. Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    return train_ds, val_ds, test_ds

#Returns cleaned DataFrame without tokenization for analysis
def get_cleaned_dataframe(data_path):
    df = pd.read_csv(data_path)
    df["cleaned_text"] = df["text"].apply(clean_text)
    df["text_type"] = df["cleaned_text"].apply(classify_text_type)
    df = df[df["text_type"].isin(["code_switched", "pure_arabic"])]
    return df[df["cleaned_text"].str.len() >= 5]


#Pipeline Execution

if __name__ == "__main__":
    import sys

    candidates = [
        Path("preprocessing/datasets/processed/unified_raw.csv"),
        Path("datasets/processed/unified_raw.csv"),
    ]
    data_path = next((p for p in candidates if p.exists()), None)

    if data_path is None:
        print("unified_raw.csv not found. Run build_dataset.py first.")
        sys.exit(1)

    #Print sample cleaning
    print("\nSample cleaning:")
    examples = [
        "وايت كثييييير هههههههه الخدمة سيئة 😡",
        "أحسن تطبيق إستخدمته والله ❤ amazing app",
        "الـ delivery was really slow وايد متأخر"]
    print(f"\n{'Original':<55}  {'Cleaned':<50}  Type")
    print("-" * 115)
    for text in examples:
        cleaned = clean_text(text)
        ttype = classify_text_type(cleaned)
        print(f"{text[:55]:<55}  {cleaned[:50]:<50}  {ttype}")

    #Generate cleaned_dataset.csv
    print(f"\nGenerating cleaned_dataset.csv from {data_path}...")
    df = get_cleaned_dataframe(data_path)
    df = assign_splits(df, seed=42)
    out_path = data_path.parent / "cleaned_dataset.csv"

    df[["cleaned_text", "label", "source", "label_source", "text_type", "split"]].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")
    print(f"\nTotal rows: {len(df):,}")
    print(f"\nLabel distribution:\n{df['label'].value_counts().to_string()}")
    print(f"\nSplit distribution:\n{df['split'].value_counts().to_string()}")
    print(f"\nText type distribution:\n{df['text_type'].value_counts().to_string()}\n")