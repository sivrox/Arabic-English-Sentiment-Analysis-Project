"""
Preprocessing for Gulf Arabic–English sentiment analysis.

- GulfArabicNormalizer: script normalization and dialect spelling
- BackTranslationAugmenter: Arabic -> English -> Arabic augmentation
- build_dataset: see build_dataset.py (or build-dataset.py) for master dataset creation
"""

from preprocessing.arabic_normalizer import GulfArabicNormalizer
from preprocessing.back_translation import (
    BackTranslationAugmenter,
    run_augmentation_pipeline,
)

__all__ = [
    "GulfArabicNormalizer",
    "BackTranslationAugmenter",
    "run_augmentation_pipeline",
]
