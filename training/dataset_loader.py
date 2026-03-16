"""
PyTorch Dataset and DataLoaders for Arabic sentiment.
Loads master_dataset.csv and returns train/val/test with tokenization.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}


class ArabicSentimentDataset(Dataset):
    """
    PyTorch Dataset for Arabic/Hinglish sentiment classification.
    Tokenizes all texts on initialization.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int = 128,
    ) -> None:
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        enc = tokenizer(
            self.texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels_tensor = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels_tensor[idx],
        }


def load_splits(
    dataset_path: str | Path,
    tokenizer,
    max_length: int = 128,
    batch_size: int = 16,
    filter_text_type: Optional[str] = None,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load master_dataset.csv and return train/val/test DataLoaders.
    filter_text_type: 'code_switched' to load only CS rows, None for all.
    Logs split sizes and class distributions.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "split" not in df.columns:
        log.warning("No 'split' column; using 80/10/10 train/val/test")
        g = df.groupby("label", group_keys=False)
        df = g.apply(lambda x: x.sample(frac=1, random_state=seed)).reset_index(drop=True)
        n = len(df)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        df["split"] = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    if filter_text_type is not None:
        if "text_type" in df.columns:
            df = df[df["text_type"] == filter_text_type].copy()
        else:
            log.warning("No text_type column; ignoring filter_text_type")
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    if "label" not in df.columns and "label_id" in df.columns:
        label_col = "label_id"
    else:
        label_col = "label"
    text_col = "text"
    for part, part_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        log.info("%s size: %d", part, len(part_df))
        if part_df[label_col].dtype == object or part_df[label_col].dtype.name == "object":
            part_df = part_df.copy()
            part_df[label_col] = part_df[label_col].map(LABEL_MAP)
        if part == "train" and len(part_df) > 0:
            vc = part_df[label_col].value_counts().sort_index()
            log.info("Train class distribution: %s", vc.to_dict())
    # Map string labels to integers before creating dataset
    if train_df[label_col].dtype == object:
        train_df = train_df.copy()
        train_df[label_col] = train_df[label_col].map(LABEL_MAP)
    train_ds = ArabicSentimentDataset(
        texts=train_df[text_col].astype(str).tolist(),
        labels=train_df[label_col].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    if val_df[label_col].dtype == object:
        val_df = val_df.copy()
        val_df[label_col] = val_df[label_col].map(LABEL_MAP)
    val_ds = ArabicSentimentDataset(
        texts=val_df[text_col].astype(str).tolist(),
        labels=val_df[label_col].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    if test_df[label_col].dtype == object:
        test_df = test_df.copy()
        test_df[label_col] = test_df[label_col].map(LABEL_MAP)
    test_ds = ArabicSentimentDataset(
        texts=test_df[text_col].astype(str).tolist(),
        labels=test_df[label_col].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def get_class_weights(train_df: pd.DataFrame, label_col: str = "label") -> torch.Tensor:
    """
    Compute inverse frequency weights for weighted CrossEntropyLoss.
    Returns tensor of shape (3,) ordered [negative, neutral, positive].
    """
    if label_col not in train_df.columns:
        return torch.ones(3)
    counts = train_df[label_col].value_counts().sort_index()
    n = len(train_df)
    weights = []
    for c in [0, 1, 2]:
        w = n / (3 * counts.get(c, 1e-6))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)
