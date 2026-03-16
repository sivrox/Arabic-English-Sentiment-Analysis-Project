"""
Training utilities for SentimentGulf.

- ArabicSentimentDataset: PyTorch Dataset with tokenization
- load_splits: train/val/test DataLoaders from master_dataset.csv
- get_class_weights: for weighted CrossEntropyLoss
- PyTorchSentimentTrainer: full training loop (raw PyTorch)
- EarlyStopping, MetricsLogger
- get_differential_optimizer, get_linear_warmup_scheduler
"""

from training.dataset_loader import (
    ArabicSentimentDataset,
    load_splits,
    get_class_weights,
    LABEL_MAP,
    ID_TO_LABEL,
)
from training.trainer_pytorch import PyTorchSentimentTrainer
from training.callbacks import EarlyStopping, MetricsLogger
from training.optimizers import get_differential_optimizer, get_linear_warmup_scheduler

__all__ = [
    "ArabicSentimentDataset",
    "load_splits",
    "get_class_weights",
    "LABEL_MAP",
    "ID_TO_LABEL",
    "PyTorchSentimentTrainer",
    "EarlyStopping",
    "MetricsLogger",
    "get_differential_optimizer",
    "get_linear_warmup_scheduler",
]
