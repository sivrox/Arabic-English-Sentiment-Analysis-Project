"""
build_dataset.py
================
Builds master_dataset.csv from all raw files in data/raw/.

Run from project root:
    python preprocessing/build_dataset.py

Outputs:
    data/processed/master_dataset.csv       <- full training pool
    data/processed/code_switched_only.csv   <- real CS samples only (for CSS metric)
    data/processed/gold_test_set.csv        <- manually-labeled real CS samples
    data/processed/build_report.txt         <- summary statistics

Files expected in data/raw/:
    appstore.csv
    arbml-Arabic_Sentiment_Twitter_Corpus_*.parquet  (any name starting with arbml)
    CompanyReviews.csv
    MagedSaeed-*.parquet                             (any name starting with MagedSaeed)
    MSA_TRAIN.txt, MSA_TEST.txt
    SA_TRAIN.txt, SA_TEST.txt
    Tweets.txt
    reddit_gulf.csv
    youtube_gulf.csv
"""

import pandas as pd
import numpy as np
import re
import logging
import unicodedata
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ARABIC SENTIMENT LEXICON
# ══════════════════════════════════════════════════════════════════════════════

# Combined lexicon built from three sources:
#   1. ArSEL  (github.com/Husseinjd/ArSEL)       — 28K+ MSA lemmas
#   2. AraSenti (github.com/nora-twairesh/AraSenti) — Saudi dialect lexicon
#   3. Hardcoded Gulf social media terms           — specific to our domain
#
# If you download the actual ArSEL CSV, set ARSEL_PATH below and it will be
# loaded and merged with the hardcoded lexicon automatically.
# Download: https://github.com/Husseinjd/ArSEL/blob/master/ArSEL.csv
#
ARSEL_PATH    = RAW / "ArSEL.csv"        # optional — script works without it
ARASENTI_PATH = RAW / "AraSenti_Lex.txt" # optional — from nora-twairesh/AraSenti

# Hardcoded lexicon — covers the most frequent sentiment words in Gulf Arabic
# social media. Scores: +1.0 = strongly positive, -1.0 = strongly negative.
HARDCODED_LEXICON = {
    # ── Strong positive ──────────────────────────────────────────
    "ممتاز":   1.0,  "رائع":    1.0,  "مميز":    1.0,  "احسن":   1.0,
    "جيد":     0.8,  "جميل":    0.8,  "زين":     0.9,  "كويس":   0.8,
    "حلو":     0.8,  "عالي":    0.7,  "ماشاءالله": 0.9, "الله يعطيك": 0.9,
    "يسلم":    0.8,  "مبدع":    0.9,  "محترم":   0.8,  "سريع":   0.6,
    "اسرع":    0.6,  "ناجح":    0.8,  "راضي":    0.7,  "سهل":    0.6,
    "مريح":    0.7,  "ممتعة":   0.8,  "افضل":    0.8,  "نظيف":   0.7,
    "لذيذ":    0.9,  "طازج":    0.7,  "ودود":    0.8,  "خبير":   0.7,
    "دقيق":    0.7,  "منظم":    0.7,
    # English positive words that Gulf users write
    "excellent": 1.0, "amazing": 1.0, "great":   0.9,  "good":   0.7,
    "perfect":   1.0, "awesome": 0.9,  "love":    0.9,  "best":   0.9,
    "fast":      0.6, "clean":   0.7,  "helpful": 0.8,  "happy":  0.9,
    "satisfied": 0.8, "nice":    0.7,  "fresh":   0.7,  "easy":   0.6,
    "quick":     0.6, "delicious": 0.9, "wonderful": 1.0, "fantastic": 1.0,
    # ── Strong negative ──────────────────────────────────────────
    "سيء":    -1.0, "سيئ":   -1.0, "سيئة":   -1.0, "رديء":  -1.0,
    "فاشل":   -1.0, "زفت":   -0.9, "خربان":  -0.9, "مو زين": -0.8,
    "بطيء":   -0.8, "بطيئ":  -0.8, "متاخر":  -0.7, "متأخر": -0.7,
    "مشكلة":  -0.7, "مشاكل": -0.7, "خطأ":    -0.6, "غلط":   -0.6,
    "محتال":  -1.0, "نصب":   -1.0, "كذب":    -0.9, "كذاب":  -0.9,
    "سرقة":   -1.0, "حرامي": -1.0, "خسارة":  -0.8, "ضيعت":  -0.7,
    "غاشم":   -0.9, "مزور":  -0.9, "احتيال": -1.0, "وايع":  -0.8,
    "مؤلم":   -0.7, "صعب":   -0.5, "معقد":   -0.5, "غالي":  -0.4,
    "ما شتغل": -0.8, "ما يشتغل": -0.8, "لا يشتغل": -0.8,
    # English negative words that Gulf users write
    "bad":      -1.0, "terrible": -1.0, "awful":  -1.0, "worst": -1.0,
    "horrible": -1.0, "useless":  -0.9, "waste":  -0.8, "slow":  -0.7,
    "broken":   -0.9, "failed":   -0.9, "scam":   -1.0, "fake":  -1.0,
    "poor":     -0.8, "never":    -0.6, "fraud":  -1.0, "rude":  -0.8,
    "disappointed": -0.9, "unacceptable": -1.0, "pathetic": -0.9,
    "disappointed": -0.9, "late":  -0.6, "missing": -0.7, "wrong": -0.7,
    "steal":    -1.0, "robbed":   -1.0, "cheated": -1.0, "lies":  -0.9,
}


def load_lexicon() -> dict:
    """
    Build the combined sentiment lexicon.
    Merges hardcoded Gulf terms with ArSEL if downloaded.

    Returns:
        dict mapping Arabic/English word -> float polarity score
        positive > +0.1, negative < -0.1, neutral in between
    """
    lexicon = HARDCODED_LEXICON.copy()

    # Load ArSEL if available
    if ARSEL_PATH.exists():
        log.info("Loading ArSEL lexicon...")
        try:
            arsel = pd.read_csv(ARSEL_PATH)
            # ArSEL columns: word, PosScore, NegScore, ObjScore (approx)
            # Net score = PosScore - NegScore
            pos_col = next((c for c in arsel.columns if 'pos' in c.lower()), None)
            neg_col = next((c for c in arsel.columns if 'neg' in c.lower()), None)
            word_col = arsel.columns[0]
            if pos_col and neg_col:
                for _, row in arsel.iterrows():
                    word = str(row[word_col]).strip()
                    score = float(row[pos_col]) - float(row[neg_col])
                    if word not in lexicon:  # don't override hardcoded Gulf terms
                        lexicon[word] = score
                log.info(f"  ArSEL: {len(arsel)} entries loaded")
        except Exception as e:
            log.warning(f"  Could not load ArSEL: {e}")

    # Load AraSenti if available
    if ARASENTI_PATH.exists():
        log.info("Loading AraSenti lexicon (Saudi dialect)...")
        try:
            with open(ARASENTI_PATH, encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        word, polarity = parts[0], parts[1].lower()
                        if word not in lexicon:
                            if polarity in ('positive', 'pos', '1'):
                                lexicon[word] = 0.7
                            elif polarity in ('negative', 'neg', '-1'):
                                lexicon[word] = -0.7
            log.info(f"  AraSenti: entries loaded")
        except Exception as e:
            log.warning(f"  Could not load AraSenti: {e}")

    log.info(f"Combined lexicon: {len(lexicon)} entries total")
    return lexicon


LEXICON = load_lexicon()


def lexicon_label(text: str, threshold: float = 0.05) -> tuple[str, float, float]:
    """
    Assign sentiment label using the combined lexicon.
    Scores each token, averages, then applies threshold.

    Args:
        text:      Input text (Arabic/English mixed)
        threshold: Minimum absolute score to assign pos/neg (else neutral)

    Returns:
        (label, score, coverage) where:
            label    = 'positive' / 'negative' / 'neutral'
            score    = mean polarity score across matched tokens
            coverage = fraction of tokens found in lexicon
    """
    tokens = str(text).lower().split()
    if not tokens:
        return 'neutral', 0.0, 0.0

    scores = []
    for token in tokens:
        # clean punctuation for lookup
        clean = re.sub(r'[^\u0600-\u06FFa-zA-Z]', '', token)
        if clean in LEXICON:
            scores.append(LEXICON[clean])

    if not scores:
        return 'neutral', 0.0, 0.0

    mean_score = np.mean(scores)
    coverage   = len(scores) / len(tokens)

    if mean_score > threshold:
        label = 'positive'
    elif mean_score < -threshold:
        label = 'negative'
    else:
        label = 'neutral'

    return label, round(float(mean_score), 4), round(float(coverage), 4)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def classify_text(text: str) -> str:
    """Classify text as code_switched / pure_arabic / pure_english / other."""
    text = str(text)
    has_arabic  = bool(re.search(r'[\u0600-\u06FF]', text))
    has_english = bool(re.search(r'[a-zA-Z]{2,}', text))
    if has_arabic and has_english:
        return 'code_switched'
    if has_arabic:
        return 'pure_arabic'
    if has_english:
        return 'pure_english'
    return 'other'


# Gulf Arabic dialect markers — used for dialect confidence scoring
GULF_MARKERS = [
    'وايد', 'زين', 'يبي', 'الحين', 'ليش', 'شلون', 'وش', 'عاد', 'لو',
    'ابشر', 'خوش', 'معليش', 'تره', 'اوكي', 'زبالة', 'شكو', 'مالت',
    'فاهم', 'عارف', 'ابد', 'هالشي', 'هالتطبيق', 'بعد', 'بس',
    'والله', 'يالله', 'ماشاء', 'طيب', 'تعال', 'روح',
]

def dialect_confidence(text: str) -> float:
    """Score 0-1 indicating likelihood this is Gulf Arabic dialect."""
    text_lower = str(text)
    hits = sum(1 for m in GULF_MARKERS if m in text_lower)
    # cap at 5 markers = full confidence
    return min(round(hits / 5.0, 2), 1.0)


def normalize_label(raw) -> str | None:
    """Map any label format to: positive / negative / neutral."""
    mapping = {
        '1': 'positive',   'pos': 'positive',  'positive': 'positive',
        'Positive': 'positive', '2': 'positive', '4': 'positive', '5': 'positive',
        '-1': 'negative',  'neg': 'negative',  'negative': 'negative',
        'Negative': 'negative', '0': 'negative',
        'neutral':  'neutral',  'Neutral': 'neutral',
        'mixed':    'neutral',  'objective': 'neutral', '3': 'neutral',
    }
    return mapping.get(str(raw).strip(), None)


def star_to_label(stars) -> str:
    """Convert 1-5 star rating to sentiment label."""
    try:
        s = int(float(str(stars)))
        if s >= 4: return 'positive'
        if s <= 2: return 'negative'
        return 'neutral'
    except:
        return None


def basic_clean(text: str) -> str:
    """Remove URLs, mentions, RT markers, extra whitespace."""
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)        # URLs
    text = re.sub(r'@\w+', '', text)                   # mentions
    text = re.sub(r'^RT\s*:?\s*', '', text)            # RT marker
    text = re.sub(r'&\w+;', '', text)                  # HTML entities
    text = re.sub(r'\s+', ' ', text).strip()           # whitespace
    return text


def make_frame(texts, labels, source, label_source, original_texts=None):
    """Build a standardized DataFrame chunk from lists."""
    df = pd.DataFrame({
        'text':          [str(t) for t in texts],
        'label':         labels,
        'source':        source,
        'label_source':  label_source,
        'original_text': original_texts if original_texts else texts,
    })
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — INDIVIDUAL DATASET LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_arbml_parquet() -> pd.DataFrame:
    """
    arbml/Arabic_Sentiment_Twitter_Corpus
    Saved as .parquet file in data/raw/
    Has manual sentiment labels.
    """
    log.info("── Loading arbml Arabic Sentiment Twitter Corpus (parquet)...")
    candidates = list(RAW.glob("arbml*.parquet")) + list(RAW.glob("Arabic_Sentiment*.parquet"))
    if not candidates:
        log.warning("   arbml parquet not found — skipping")
        return pd.DataFrame()

    frames = []
    for path in candidates:
        df = pd.read_parquet(path)
        log.info(f"   Loaded {path.name}: {len(df)} rows | columns: {list(df.columns)}")

        # find text + label columns by keyword
        text_col  = next((c for c in df.columns if any(k in c.lower() for k in ['text','tweet','content'])), df.columns[0])
        label_col = next((c for c in df.columns if any(k in c.lower() for k in ['label','sentiment','class','polarity'])), None)

        if label_col is None:
            log.warning(f"   No label column found in {path.name} — using lexicon labels")
            sub = df[[text_col]].copy()
            sub.columns = ['text']
            sub['label']        = sub['text'].apply(lambda t: lexicon_label(t)[0])
            sub['label_source'] = 'weak_lexicon'
        else:
            sub = df[[text_col, label_col]].copy()
            sub.columns = ['text', 'raw_label']
            sub['label']        = sub['raw_label'].apply(normalize_label)
            sub['label_source'] = 'manual'

        sub['source']        = 'arbml_twitter'
        sub['original_text'] = sub['text']
        frames.append(sub[['text','label','source','label_source','original_text']])

    result = pd.concat(frames, ignore_index=True)
    log.info(f"   arbml total: {len(result)} rows")
    return result


def load_magedsaeed_parquet() -> pd.DataFrame:
    """
    MagedSaeed/arabic-english-code-switching-text
    Saved as .parquet. Text only — no labels.
    MOST IMPORTANT: real code-switched samples.
    """
    log.info("── Loading MagedSaeed CS dataset (parquet)...")
    candidates = list(RAW.glob("MagedSaeed*.parquet")) + list(RAW.glob("arabic-english*.parquet"))
    if not candidates:
        log.warning("   MagedSaeed parquet not found — skipping")
        return pd.DataFrame()

    frames = []
    for path in candidates:
        df = pd.read_parquet(path)
        log.info(f"   Loaded {path.name}: {len(df)} rows | columns: {list(df.columns)}")
        text_col = df.columns[0]
        sub = df[[text_col]].copy()
        sub.columns = ['text']

        # assign weak labels + coverage score
        results = sub['text'].apply(lambda t: lexicon_label(t))
        sub['label']            = results.apply(lambda r: r[0])
        sub['lexicon_score']    = results.apply(lambda r: r[1])
        sub['lexicon_coverage'] = results.apply(lambda r: r[2])

        # drop rows with zero lexicon coverage — label is pure noise
        before = len(sub)
        sub = sub[sub['lexicon_coverage'] > 0.0]
        log.info(f"   Coverage filter: dropped {before - len(sub)} zero-coverage rows")

        sub['source']        = 'magedsaeed_cs'
        sub['label_source']  = 'weak_lexicon'
        sub['original_text'] = sub['text']
        frames.append(sub[['text','label','source','label_source','original_text']])

    result = pd.concat(frames, ignore_index=True)
    log.info(f"   MagedSaeed total: {len(result)} rows")
    return result


def load_astd_tweets() -> pd.DataFrame:
    """
    mahmoudnabil/ASTD — Tweets.txt
    Format: one tweet per line, label is first character or tab-separated.
    """
    log.info("── Loading ASTD Tweets.txt...")
    path = RAW / "Tweets.txt"
    if not path.exists():
        log.warning("   Tweets.txt not found — skipping")
        return pd.DataFrame()

    rows = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try tab-separated first
            if '\t' in line:
                parts = line.split('\t', 1)
                label_raw, text = parts[0].strip(), parts[1].strip()
            else:
                # fallback: first token is label
                parts = line.split(' ', 1)
                label_raw = parts[0].strip()
                text      = parts[1].strip() if len(parts) > 1 else ''
            rows.append({'text': text, 'raw_label': label_raw})

    df = pd.DataFrame(rows)
    df['label']        = df['raw_label'].apply(normalize_label)
    df['source']       = 'astd_tweets'
    df['label_source'] = 'manual'
    df['original_text'] = df['text']
    log.info(f"   ASTD: {len(df)} rows loaded")
    return df[['text','label','source','label_source','original_text']]


def load_company_reviews() -> pd.DataFrame:
    """
    CompanyReviews.csv — columns: review_description, rating, company
    Ratings: -1 / 0 / 1 directly mapped to neg/neu/pos.
    """
    log.info("── Loading CompanyReviews.csv...")
    path = RAW / "CompanyReviews.csv"
    if not path.exists():
        log.warning("   CompanyReviews.csv not found — skipping")
        return pd.DataFrame()

    df = pd.read_csv(path, encoding='utf-8-sig')
    log.info(f"   Loaded: {len(df)} rows | columns: {list(df.columns)}")

    df = df[['review_description', 'rating']].copy()
    df.columns = ['text', 'raw_label']
    df['label']         = df['raw_label'].apply(normalize_label)
    df['source']        = 'company_reviews'
    df['label_source']  = 'manual'
    df['original_text'] = df['text']
    log.info(f"   CompanyReviews: {len(df)} rows loaded")
    return df[['text','label','source','label_source','original_text']]


def load_appstore() -> pd.DataFrame:
    """
    appstore.csv — scraped UAE Google Play reviews.
    Has star ratings as weak labels.
    """
    log.info("── Loading appstore.csv...")
    path = RAW / "appstore.csv"
    if not path.exists():
        log.warning("   appstore.csv not found — skipping")
        return pd.DataFrame()

    df = pd.read_csv(path, encoding='utf-8-sig')
    log.info(f"   Loaded: {len(df)} rows | columns: {list(df.columns)}")

    # find text column
    text_col = next((c for c in df.columns if any(k in c.lower() for k in ['text','review','content','comment'])), df.columns[0])

    # find label/stars column
    star_col  = next((c for c in df.columns if any(k in c.lower() for k in ['star','rating','score'])), None)
    label_col = next((c for c in df.columns if any(k in c.lower() for k in ['label','sentiment'])), None)

    sub = df[[text_col]].copy()
    sub.columns = ['text']

    if label_col:
        sub['label']        = df[label_col].apply(normalize_label)
        sub['label_source'] = 'weak_stars'
    elif star_col:
        sub['label']        = df[star_col].apply(star_to_label)
        sub['label_source'] = 'weak_stars'
    else:
        sub['label']        = sub['text'].apply(lambda t: lexicon_label(t)[0])
        sub['label_source'] = 'weak_lexicon'

    sub['source']        = 'appstore_scrape'
    sub['original_text'] = sub['text']
    log.info(f"   Appstore: {len(sub)} rows loaded")
    return sub[['text','label','source','label_source','original_text']]


def load_reddit() -> pd.DataFrame:
    """
    reddit_gulf.csv — scraped from r/dubai, r/UAE etc.
    No inherent labels — use lexicon.
    """
    log.info("── Loading reddit_gulf.csv...")
    path = RAW / "reddit_gulf.csv"
    if not path.exists():
        log.warning("   reddit_gulf.csv not found — skipping")
        return pd.DataFrame()

    df = pd.read_csv(path, encoding='utf-8-sig')
    log.info(f"   Loaded: {len(df)} rows | columns: {list(df.columns)}")

    text_col = next((c for c in df.columns if any(k in c.lower() for k in ['text','post','comment','body','content'])), df.columns[0])
    sub = df[[text_col]].copy()
    sub.columns = ['text']

    results = sub['text'].apply(lambda t: lexicon_label(t))
    sub['label']            = results.apply(lambda r: r[0])
    sub['lexicon_coverage'] = results.apply(lambda r: r[2])

    # drop zero-coverage rows (cannot reliably label)
    before = len(sub)
    sub = sub[sub['lexicon_coverage'] > 0.0]
    log.info(f"   Reddit coverage filter: dropped {before - len(sub)} rows")

    sub['source']        = 'reddit_gulf'
    sub['label_source']  = 'weak_lexicon'
    sub['original_text'] = sub['text']
    log.info(f"   Reddit: {len(sub)} rows loaded")
    return sub[['text','label','source','label_source','original_text']]


def load_youtube() -> pd.DataFrame:
    """
    youtube_gulf.csv — scraped UAE YouTube comments.
    NO LABELS. These go to gold test set only after manual annotation.
    We include them here with weak labels BUT flag them separately.
    """
    log.info("── Loading youtube_gulf.csv (flagged for manual annotation)...")
    path = RAW / "youtube_gulf.csv"
    if not path.exists():
        log.warning("   youtube_gulf.csv not found — skipping")
        return pd.DataFrame()

    df = pd.read_csv(path, encoding='utf-8-sig')
    log.info(f"   Loaded: {len(df)} rows | columns: {list(df.columns)}")

    text_col = next((c for c in df.columns if any(k in c.lower() for k in ['text','comment','content'])), df.columns[0])
    sub = df[[text_col]].copy()
    sub.columns = ['text']

    results = sub['text'].apply(lambda t: lexicon_label(t))
    sub['label']           = results.apply(lambda r: r[0])
    sub['label_source']    = 'weak_lexicon_NEEDS_MANUAL'  # flagged!
    sub['source']          = 'youtube_scrape'
    sub['original_text']   = sub['text']

    log.info(f"   YouTube: {len(sub)} rows — NEEDS manual annotation before use in test set")
    return sub[['text','label','source','label_source','original_text']]


def load_are_cstd_txt(filename: str, source_tag: str) -> pd.DataFrame:
    """
    ArE-CSTD plain text files (SA_TRAIN, SA_TEST, MSA_TRAIN, MSA_TEST).
    One code-switched sentence per line. No labels — use lexicon.
    """
    log.info(f"── Loading {filename}...")
    path = RAW / filename
    if not path.exists():
        log.warning(f"   {filename} not found — skipping")
        return pd.DataFrame()

    lines = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    log.info(f"   {filename}: {len(lines)} lines read")

    sub = pd.DataFrame({'text': lines})
    results = sub['text'].apply(lambda t: lexicon_label(t))
    sub['label']            = results.apply(lambda r: r[0])
    sub['lexicon_coverage'] = results.apply(lambda r: r[2])

    # for synthetic data, apply stricter coverage filter
    before = len(sub)
    sub = sub[sub['lexicon_coverage'] > 0.05]
    log.info(f"   Coverage filter: {before - len(sub)} rows dropped")

    sub['source']        = source_tag
    sub['label_source']  = 'weak_lexicon_synthetic'
    sub['original_text'] = sub['text']
    return sub[['text','label','source','label_source','original_text']]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MAIN BUILD PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_master_dataset():

    log.info("=" * 65)
    log.info("  SentimentGulf — Master Dataset Build")
    log.info("=" * 65)

    # ── Load all sources ────────────────────────────────────────────────────
    frames = [
        load_astd_tweets(),
        load_company_reviews(),
        load_appstore(),
        load_reddit(),
        load_youtube(),
        load_arbml_parquet(),
        load_magedsaeed_parquet(),
        load_are_cstd_txt("SA_TRAIN.txt",  "are_cstd_saudi_train"),
        load_are_cstd_txt("SA_TEST.txt",   "are_cstd_saudi_test"),
        load_are_cstd_txt("MSA_TRAIN.txt", "are_cstd_msa_train"),
        load_are_cstd_txt("MSA_TEST.txt",  "are_cstd_msa_test"),
    ]

    # drop empty frames
    frames = [f for f in frames if not f.empty]

    # ── Merge ───────────────────────────────────────────────────────────────
    log.info("\n── Merging all frames...")
    df = pd.concat(frames, ignore_index=True)
    log.info(f"   Raw merged total: {len(df):,} rows")

    # ── Clean ───────────────────────────────────────────────────────────────
    log.info("── Cleaning...")
    df['text']          = df['text'].astype(str).str.strip()
    df['original_text'] = df['original_text'].astype(str).str.strip()
    df['text']          = df['text'].apply(basic_clean)

    # drop bad rows
    df = df[df['text'].str.len() > 10]       # too short
    df = df[df['text'] != 'nan']             # empty
    df = df[df['label'].notna()]             # unrecognized label
    df = df[df['label'] != 'None']

    # drop pure English — irrelevant to the task
    df['text_type'] = df['text'].apply(classify_text)
    before = len(df)
    df = df[df['text_type'].isin(['pure_arabic', 'code_switched'])]
    log.info(f"   Dropped {before - len(df)} pure-English / other rows")

    # deduplicate on cleaned text
    before = len(df)
    df.drop_duplicates(subset='text', inplace=True)
    log.info(f"   Deduplication removed: {before - len(df)} rows")

    # ── Add metadata columns ────────────────────────────────────────────────
    log.info("── Computing metadata columns...")
    df['dialect_confidence'] = df['text'].apply(dialect_confidence)
    df['char_length']        = df['text'].str.len()
    df['token_count']        = df['text'].str.split().str.len()

    # ── Assign train/val/test splits ────────────────────────────────────────
    log.info("── Assigning splits...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    # YouTube samples → gold test set (need manual labels)
    df.loc[df['source'] == 'youtube_scrape', 'split'] = 'gold_test_pending'

    # real code-switched non-youtube → reserve 20% for test
    real_cs_mask = (
        (df['text_type'] == 'code_switched') &
        (df['source'].isin(['company_reviews', 'appstore_scrape',
                            'magedsaeed_cs', 'reddit_gulf'])) &
        (df['split'].isna())
    )
    real_cs_idx = df[real_cs_mask].index
    test_cs_idx = real_cs_idx[:int(len(real_cs_idx) * 0.2)]
    df.loc[test_cs_idx, 'split'] = 'test'

    # remaining rows → 80/10/10 train/val/test
    remaining = df[df['split'].isna()].index
    n = len(remaining)
    train_end = int(n * 0.80)
    val_end   = int(n * 0.90)
    df.loc[remaining[:train_end],       'split'] = 'train'
    df.loc[remaining[train_end:val_end], 'split'] = 'val'
    df.loc[remaining[val_end:],          'split'] = 'test'

    # ── Report ──────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  FINAL MASTER DATASET SUMMARY")
    log.info("=" * 65)
    log.info(f"\n  Total rows:  {len(df):,}")

    log.info("\n  By label:")
    for label, count in df['label'].value_counts().items():
        pct = count / len(df) * 100
        log.info(f"    {label:<12} {count:>7,}  ({pct:.1f}%)")

    log.info("\n  By text type:")
    for tt, count in df['text_type'].value_counts().items():
        pct = count / len(df) * 100
        log.info(f"    {tt:<20} {count:>7,}  ({pct:.1f}%)")

    log.info("\n  By source:")
    for src, count in df['source'].value_counts().items():
        log.info(f"    {src:<35} {count:>7,}")

    log.info("\n  By split:")
    for split, count in df['split'].value_counts().items():
        log.info(f"    {split:<25} {count:>7,}")

    log.info("\n  By label source:")
    for ls, count in df['label_source'].value_counts().items():
        log.info(f"    {ls:<40} {count:>7,}")

    # ── Save ────────────────────────────────────────────────────────────────
    final_cols = [
        'text', 'label', 'source', 'label_source',
        'text_type', 'dialect_confidence', 'split',
        'char_length', 'token_count', 'original_text'
    ]
    df = df[final_cols]

    # 1. Full master dataset
    master_path = PROCESSED / "master_dataset.csv"
    df.to_csv(master_path, index=False, encoding='utf-8-sig')
    log.info(f"\n  Saved: {master_path}  ({len(df):,} rows)")

    # 2. Code-switched only subset (for CSS metric evaluation)
    cs_path = PROCESSED / "code_switched_only.csv"
    cs_df = df[df['text_type'] == 'code_switched']
    cs_df.to_csv(cs_path, index=False, encoding='utf-8-sig')
    log.info(f"  Saved: {cs_path}  ({len(cs_df):,} rows)")

    # 3. Gold test set (YouTube + real CS test split)
    gold_path = PROCESSED / "gold_test_set.csv"
    gold_df = df[df['split'].isin(['gold_test_pending', 'test']) &
                 (df['text_type'] == 'code_switched')]
    gold_df.to_csv(gold_path, index=False, encoding='utf-8-sig')
    log.info(f"  Saved: {gold_path}  ({len(gold_df):,} rows)")

    # 4. Text report
    report_path = PROCESSED / "build_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("SentimentGulf — Dataset Build Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total rows: {len(df):,}\n\n")
        f.write("Label distribution:\n")
        f.write(df['label'].value_counts().to_string() + "\n\n")
        f.write("Text type distribution:\n")
        f.write(df['text_type'].value_counts().to_string() + "\n\n")
        f.write("Source distribution:\n")
        f.write(df['source'].value_counts().to_string() + "\n\n")
        f.write("Split distribution:\n")
        f.write(df['split'].value_counts().to_string() + "\n\n")
    log.info(f"  Saved: {report_path}")

    log.info("\n  Build complete.")
    return df


if __name__ == "__main__":
    build_master_dataset()