"""
Back-translation augmentation for Arabic sentiment data.

Pipeline:
  Arabic text -> translate to English -> translate back to Arabic
  -> quality filter -> keep if similar enough to original

Used to augment underrepresented classes (primarily neutral)
in the training set.

Library: deep_translator (pip install deep_translator)
No API key required for Google Translate backend.
"""

import logging
import re
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None  # type: ignore


class BackTranslationAugmenter:
    """
    Augments Arabic text via English pivot translation.

    Arabic -> English -> Arabic produces paraphrased variants
    that preserve sentiment while adding lexical diversity.
    Used specifically for underrepresented classes.
    """

    def __init__(
        self,
        source_lang: str = "ar",
        pivot_lang: str = "en",
        quality_threshold: float = 0.6,
        sleep_between_requests: float = 0.5,
    ):
        """
        Args:
          source_lang: Source language code ('ar' for Arabic)
          pivot_lang: Pivot language for translation ('en')
          quality_threshold: Minimum character similarity 0-1.
            Texts scoring below this are rejected as too distorted.
            0.6 means augmented text must share 60% char n-grams
            with original.
          sleep_between_requests: Seconds to wait between API calls.
            Prevents rate limiting from Google Translate.
        """
        self.source_lang = source_lang
        self.pivot_lang = pivot_lang
        self.quality_threshold = quality_threshold
        self.sleep_between_requests = sleep_between_requests
        if GoogleTranslator is None:
            log.warning("deep_translator not installed; back-translation will fail.")

    def _char_ngram_similarity(
        self,
        text1: str,
        text2: str,
        n: int = 3,
    ) -> float:
        """
        Compute character n-gram overlap between two strings.
        Used as quality filter — rejects augmented texts that
        differ too much from the original (meaning changed).

        Returns float 0-1 (1 = identical, 0 = no overlap).
        """
        if not text1 or not text2:
            return 0.0
        text1 = text1.strip()
        text2 = text2.strip()
        if not text1:
            return 0.0
        ngrams1 = set()
        for i in range(len(text1) - n + 1):
            ngrams1.add(text1[i : i + n])
        if not ngrams1:
            return 1.0 if text1 == text2 else 0.0
        ngrams2 = set()
        for i in range(len(text2) - n + 1):
            ngrams2.add(text2[i : i + n])
        overlap = len(ngrams1 & ngrams2) / len(ngrams1)
        return min(1.0, overlap)

    def translate_single(self, text: str) -> str | None:
        """
        Translate one text: Arabic -> English -> Arabic.
        Returns augmented Arabic text or None if translation failed.

        Handles:
          - Network errors (return None, log warning)
          - Empty responses (return None)
          - Rate limit errors (sleep and retry once)
        """
        if GoogleTranslator is None:
            log.warning("deep_translator not available")
            return None
        text = (text or "").strip()
        if not text:
            return None
        try:
            time.sleep(self.sleep_between_requests)
            to_en = GoogleTranslator(source=self.source_lang, target=self.pivot_lang)
            en_text = to_en.translate(text)
            if not en_text or not str(en_text).strip():
                return None
            time.sleep(self.sleep_between_requests)
            to_ar = GoogleTranslator(source=self.pivot_lang, target=self.source_lang)
            ar_text = to_ar.translate(str(en_text))
            if ar_text is None:
                return None
            ar_text = str(ar_text).strip()
            return ar_text if ar_text else None
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(self.sleep_between_requests * 4)
                try:
                    to_en = GoogleTranslator(source=self.source_lang, target=self.pivot_lang)
                    en_text = to_en.translate(text)
                    if not en_text or not str(en_text).strip():
                        return None
                    time.sleep(self.sleep_between_requests)
                    to_ar = GoogleTranslator(source=self.pivot_lang, target=self.source_lang)
                    ar_text = to_ar.translate(str(en_text))
                    return str(ar_text).strip() if ar_text else None
                except Exception as e2:
                    log.warning("Back-translation retry failed: %s", e2)
                    return None
            log.warning("Back-translation failed for text (len=%d): %s", len(text), e)
            return None

    def quality_filter(
        self,
        original: str,
        augmented: str,
    ) -> bool:
        """
        Returns True if augmented text passes quality check.
        Rejects if:
          - Similarity below self.quality_threshold
          - Augmented text is empty
          - Augmented text is identical to original (no change)
          - Augmented text lost all Arabic characters
        """
        if not augmented or not augmented.strip():
            return False
        if original.strip() == augmented.strip():
            return False
        sim = self._char_ngram_similarity(original, augmented)
        if sim < self.quality_threshold:
            return False
        # Check that augmented still has Arabic (U+0600–U+06FF)
        if not re.search(r"[\u0600-\u06FF]", augmented):
            return False
        return True

    def augment_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label",
        target_classes: list[str] | list[int] | None = None,
        max_samples: int = 5000,
    ) -> pd.DataFrame:
        """
        Augment a DataFrame by back-translating samples from
        underrepresented classes.

        Process:
          1. Filter to target_classes rows
          2. Randomly sample up to max_samples rows
          3. For each: translate, quality check, keep if passes
          4. Mark augmented rows with source='back_translation'
             and label_source='augmented'
          5. Return DataFrame of new augmented rows only
             (caller concatenates with original)

        Args:
          df: Input DataFrame with text and label columns
          text_col: Name of text column
          label_col: Name of label column
          target_classes: List of label values to augment.
            Default: ['neutral'] (most underrepresented)
          max_samples: Maximum augmented samples to produce.
            Hard cap to prevent runaway API usage.

        Returns:
          DataFrame of augmented rows with same schema as input.
          Does NOT include original rows.

        Logs:
          - Progress every 100 samples
          - Final: attempted, passed quality, rejected counts
        """
        if target_classes is None:
            target_classes = ["neutral"]
        subset = df[df[label_col].isin(target_classes)].copy()
        if subset.empty:
            log.info("No rows in target_classes; returning empty DataFrame")
            return pd.DataFrame(columns=df.columns)
        n_take = min(len(subset), max_samples)
        subset = subset.sample(n=n_take, random_state=42).reset_index(drop=True)
        rows = []
        attempted = 0
        passed = 0
        rejected = 0
        for i, row in subset.iterrows():
            attempted += 1
            if attempted % 100 == 0:
                log.info("Back-translation progress: attempted=%d, passed=%d", attempted, passed)
            text = row[text_col]
            if pd.isna(text) or not str(text).strip():
                rejected += 1
                continue
            aug = self.translate_single(str(text))
            if aug is None:
                rejected += 1
                continue
            if not self.quality_filter(str(text), aug):
                rejected += 1
                continue
            passed += 1
            new_row = row.to_dict()
            new_row[text_col] = aug
            new_row["source"] = "back_translation"
            new_row["label_source"] = "augmented"
            if "original_text" not in new_row:
                new_row["original_text"] = str(text)
            rows.append(new_row)
            if len(rows) >= max_samples:
                break
        log.info(
            "Back-translation done: attempted=%d, passed=%d, rejected=%d",
            attempted,
            passed,
            rejected,
        )
        if not rows:
            return pd.DataFrame(columns=df.columns)
        out = pd.DataFrame(rows)
        for c in df.columns:
            if c not in out.columns:
                out[c] = None
        return out[list(df.columns)]

    def save_augmented(self, augmented_df: pd.DataFrame, output_path: Path) -> None:
        """Save augmented DataFrame to CSV with utf-8-sig encoding."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        augmented_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def run_augmentation_pipeline(
    master_dataset_path: Path,
    output_path: Path,
    config: dict,
) -> pd.DataFrame:
    """
    End-to-end augmentation pipeline.
    Loads training split, augments neutral class,
    saves augmented rows separately.

    Called from notebooks when augmentation is enabled in config.

    Returns combined DataFrame (original train + augmented).
    """
    master_dataset_path = Path(master_dataset_path)
    output_path = Path(output_path)
    if not master_dataset_path.exists():
        log.warning("Master dataset not found at %s", master_dataset_path)
        return pd.DataFrame()
    df = pd.read_csv(master_dataset_path)
    if "split" not in df.columns:
        log.warning("No 'split' column; using full dataset for augmentation")
        train_df = df
    else:
        train_df = df[df["split"] == "train"].copy()
    aug_cfg = config.get("augmentation", {})
    if not aug_cfg.get("back_translation_enabled", True):
        log.info("Back-translation disabled in config")
        return train_df
    target = aug_cfg.get("target_minority_classes", ["neutral"])
    max_aug = aug_cfg.get("max_augmented_samples", 5000)
    quality = aug_cfg.get("quality_threshold", 0.6)
    augmenter = BackTranslationAugmenter(
        source_lang=aug_cfg.get("source_lang", "ar"),
        pivot_lang=aug_cfg.get("pivot_lang", "en"),
        quality_threshold=quality,
    )
    augmented = augmenter.augment_dataframe(
        train_df,
        text_col="text",
        label_col="label",
        target_classes=target,
        max_samples=max_aug,
    )
    if not augmented.empty:
        augmenter.save_augmented(augmented, output_path)
    combined = pd.concat([train_df, augmented], ignore_index=True) if not augmented.empty else train_df
    return combined
