#DATASET BUILDER FOR 3-CLASS SENTIMENT ANALYSIS FROM SOURCED DATASETS

import re
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

#Path definitions
raw_dir = Path("preprocessing/datasets/raw")
processed_dir = Path("preprocessing/datasets/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

output_file = processed_dir / "unified_raw.csv"
arasenti_path = Path("preprocessing/datasets/lexicon/AraSentiLexiconV1.0")

#Label normalization
#Maps all raw label variants to: positive / negative / neutral

label_map = {
    "1": "positive", "pos": "positive", "positive": "positive",
    "Positive": "positive", "2": "positive", "4": "positive", "5": "positive",
    "-1": "negative", "neg": "negative", "negative": "negative",
    "Negative": "negative", "0": "negative",
    "neutral": "neutral", "Neutral": "neutral", "mixed": "neutral",
    "objective": "neutral", "3": "neutral",
}

def normalize_label(raw):
    return label_map.get(str(raw).strip(), None) #Returns None if label is unrecognized


def star_to_label(stars):
    try:
        s = int(float(str(stars)))  #Converts a 1-5 star rating to a sentiment label
        if s >= 4: return "positive"
        if s <= 2: return "negative"
        return "neutral"
    except (ValueError, TypeError):
        return None

#Lexicon Loading (ArSenti)
#Used for weak labeling on datasets with no human-annotated labels.

gulf_overrides = {
    "وايد زين": 5.0, "زين": 4.5, "وايد": 0.0, "خوش": 4.0, "يسلم": 4.0,
    "مو زين": -4.0, "زفت": -4.5, "خربان": -4.0, "فاشل": -5.0,
    "excellent": 5.0, "amazing": 5.0, "great": 4.0, "good": 3.0,
    "bad": -4.5, "terrible": -5.0, "awful": -5.0, "worst": -5.0,
    "horrible": -5.0, "slow": -3.0, "broken": -4.0, "scam": -5.0,}

def _load_lexicon():
    lex = {}
    if arasenti_path.exists():
        try:
            with open(arasenti_path, "rb") as f:
                content = f.read().decode("cp1256")  #cp1256 = Windows Arabic encoding
            for line in content.split("\n")[70:]:  #first 69 lines are header
                line = line.strip()
                if not line: continue
                parts = re.split(r"\s{2,}|\t", line)
                if len(parts) >= 2:
                    try:
                        lex[parts[0].strip()] = float(parts[-1].strip())
                    except ValueError:
                        pass
            print(f"\nAraSenti lexicon loaded: {len(lex)} entries")
        except Exception as e:
            print(f"AraSenti load failed: {e}")
    lex.update(gulf_overrides)
    return lex

lexicon = _load_lexicon() #Load lexicon once at module level to avoid reloading for every row

def lexicon_label(text):
    tokens = str(text).lower().split()      #assigns a sentiment label by averaging scores of known words in the text
    scores = [lexicon[re.sub(r"[^\u0600-\u06FFa-zA-Z]", "", t)]
              for t in tokens if re.sub(r"[^\u0600-\u06FFa-zA-Z]", "", t) in lexicon]
    if not scores: return "neutral"
    mean = float(np.mean(scores))
    if mean > 0.5: return "positive"
    if mean < -0.5: return "negative"
    return "neutral"

#Helper functions for Column Detection

def _find(pattern):
    matches = list(raw_dir.glob(pattern)) #Searches raw_dir for a file matching the given glob pattern
    return matches[0] if matches else None

def _detect_text_col(df):
    for kw in ["text", "tweet", "content", "review", "comment", "post", "body"]: #Returns the name of the text column based on common keywords
        match = next((c for c in df.columns if kw in c.lower()), None)
        if match: return match
    return df.columns[0]

def _detect_label_col(df):
    for kw in ["label", "sentiment", "class", "polarity"]:
        match = next((c for c in df.columns if kw in c.lower()), None)
        if match: return match
    return None

def _detect_star_col(df):
    for kw in ["star", "rating", "score"]:
        match = next((c for c in df.columns if kw in c.lower()), None)
        if match: return match
    return None

#Dataset loaders
#Each function loads one source of data and returns a DataFrame with four columns: text (raw), label, source, label_source

#ASTD: Arabic Sentiment Tweets Dataset
def load_astd():
    path = raw_dir / "Tweets.txt"
    if not path.exists():
        print("Astd: file not found, skipping.")
        return pd.DataFrame()
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
    print(f"Astd loaded: {len(df)} rows")
    return df[["text", "label", "source", "label_source"]]

#Arbml Arabic Sentiment Twitter Corpus
def load_arbml():
    path = _find("arbml*.parquet") or _find("Arabic_Sentiment*.parquet")
    if not path:
        print("Arbml: file not found, skipping.")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    text_col = _detect_text_col(df)
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
    print(f"Arbml loaded: {len(out)} rows")
    return out[["text", "label", "source", "label_source"]]

#MagedSaeed Arabic-English code-switching dataset (Primary CS Source)
def load_magedsaeed():
    path = _find("MagedSaeed*.parquet") or _find("arabic-english*.parquet")
    if not path:
        print("Magedsaeed: file not found, skipping.")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    out = df[[df.columns[0]]].copy()
    out.columns = ["text"]
    out["label"] = out["text"].apply(lexicon_label)
    out["source"] = "magedsaeed_cs"
    out["label_source"] = "weak_lexicon"
    print(f"Magedsaeed loaded: {len(out)} rows")
    return out[["text", "label", "source", "label_source"]]

#Company reviews from Kaggle (Talabat, Careem...)
def load_company_reviews():
    path = raw_dir / "CompanyReviews.csv"
    if not path.exists():
        print("Company reviews: file not found, skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    out = df[["review_description", "rating"]].copy()
    out.columns = ["text", "raw_label"]
    out["label"] = out["raw_label"].apply(normalize_label)
    out["source"] = "company_reviews"
    out["label_source"] = "weak_stars"
    print(f"Company reviews loaded: {len(out)} rows")
    return out[["text", "label", "source", "label_source"]]

#Scraped UAE app store reviews
def load_appstore():
    path = raw_dir / "appstore.csv"
    if not path.exists():
        print("Appstore: file not found, skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    text_col = _detect_text_col(df)
    star_col = _detect_star_col(df)
    label_col = _detect_label_col(df)
    out = df[[text_col]].copy()
    out.columns = ["text"]
    if label_col:
        out["label"] = df[label_col].apply(normalize_label)
        out["label_source"] = "manual"
    elif star_col:
        out["label"] = df[star_col].apply(star_to_label)
        out["label_source"] = "weak_stars"
    else:
        out["label"] = out["text"].apply(lexicon_label)
        out["label_source"] = "weak_lexicon"
    out["source"] = "appstore_scrape"
    print(f"Appstore loaded: {len(out)} rows")
    return out[["text", "label", "source", "label_source"]]

#Scraped Reddit comments from Gulf-related subreddits
def load_reddit():
    path = raw_dir / "reddit_gulf.csv"
    if not path.exists():
        print("Reddit: file not found, skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    text_col = _detect_text_col(df)
    out = df[[text_col]].copy()
    out.columns = ["text"]
    out["label"] = out["text"].apply(lexicon_label)
    out["source"] = "reddit_gulf"
    out["label_source"] = "weak_lexicon"
    print(f"Reddit loaded: {len(out)} rows")
    return out[["text", "label", "source", "label_source"]]

#Scraped YouTube comments from Gulf Arabic channels - assigned to gold_test split in preprocessor.py due to lower label reliability and will not be used for training
def load_youtube():
    path = raw_dir / "youtube_gulf.csv"
    if not path.exists():
        print("Youtube: file not found, skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    text_col = _detect_text_col(df)
    out = df[[text_col]].copy()
    out.columns = ["text"]
    out["label"] = out["text"].apply(lexicon_label)
    out["source"] = "youtube_scrape"
    out["label_source"] = "weak_lexicon_needs_manual"
    print(f"Youtube loaded: {len(out)} rows")
    return out[["text", "label", "source", "label_source"]]

#ArE-CSTD: Arabic-English Code-Switching Textual Dataset (Synthetic augmentation)
def load_are_cstd(filename, source_tag):
    path = raw_dir / filename
    if not path.exists():
        print(f"{filename}: file not found, skipping.")
        return pd.DataFrame()
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [l.strip() for l in f if l.strip()]
    out = pd.DataFrame({"text": lines})
    out["label"] = out["text"].apply(lexicon_label)
    out["source"] = source_tag
    out["label_source"] = "weak_lexicon_synthetic"
    print(f"{filename} loaded: {len(out)} rows")
    return out[["text", "label", "source", "label_source"]]


#Build function
def build_unified_dataset():
    print("\nBuilding unified raw dataset...")
    frames = [
        load_astd(), load_arbml(), load_magedsaeed(),
        load_company_reviews(), load_appstore(), load_reddit(), load_youtube(),
        load_are_cstd("SA_TRAIN.txt", "are_cstd_saudi_train"),
        load_are_cstd("SA_TEST.txt", "are_cstd_saudi_test"),
        load_are_cstd("MSA_TRAIN.txt", "are_cstd_msa_train"),
        load_are_cstd("MSA_TEST.txt", "are_cstd_msa_test"),]
    frames = [f for f in frames if not f.empty]

    if not frames:
        print("\nNo data sources found. Check that raw files exist in preprocessing/datasets/raw/")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows after merge: {len(df)}")

    #Basic filtering
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 3] #drop near-empty rows
    df = df[df["text"] != "nan"] #drop stringified nulls
    df = df[df["label"].notna()] #drop unrecognized labels
    df = df[df["label"] != "None"]

    #Keep only rows with Arabic or English characters
    has_arabic = df["text"].str.contains(r"[\u0600-\u06FF]", regex=True, na=False)
    has_english = df["text"].str.contains(r"[a-zA-Z]{2,}", regex=True, na=False)
    df = df[has_arabic | has_english]

    #Remove duplicate texts
    before = len(df)
    df.drop_duplicates(subset="text", inplace=True)
    print(f"\nDuplicates removed: {before - len(df)}")

    #Shuffle rows so sources are mixed before splitting in preprocessor.py
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nFinal row count: {len(df)}")
    print(f"\nLabel distribution:\n{df['label'].value_counts().to_string()}")
    print(f"\nSource distribution:\n{df['source'].value_counts().to_string()}")

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved to: {output_file}")

if __name__ == "__main__":
    build_unified_dataset()