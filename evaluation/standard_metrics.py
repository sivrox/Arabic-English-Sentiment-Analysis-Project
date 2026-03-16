"""
Standard evaluation metrics for SentimentGulf.

Metrics computed:
  - Accuracy
  - Macro-averaged F1 (PRIMARY metric)
  - Per-class F1, Precision, Recall
  - BLEU score (required by project spec)
  - Matthews Correlation Coefficient (MCC)
  - Confusion matrix

BLEU note: For classification tasks, BLEU is computed by
treating predicted label sequences as 'generated text'
and true label sequences as 'reference text'. This is
a non-standard use of BLEU but satisfies the project
requirement to include it as a reported metric.
The more meaningful interpretation uses BLEU on any
free-text generation outputs if produced.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

log = logging.getLogger(__name__)

try:
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
    nltk.download("punkt", quiet=True)
except Exception as e:
    log.warning("nltk/bleu not available: %s", e)
    corpus_bleu = None
    sentence_bleu = None
    SmoothingFunction = None


class SentimentEvaluator:
    LABEL_NAMES = ["negative", "neutral", "positive"]

    def compute_all_metrics(
        self,
        y_true: list[int],
        y_pred: list[int],
        y_prob: Optional[list[list[float]]] = None,
        prefix: str = "",
    ) -> dict[str, Any]:
        """
        Full metric suite.
        BLEU computed on label-sequence level:
          references = [[str(t)] for t in y_true]
          hypotheses = [str(p) for p in y_pred]
        Returns dict with all metrics prefixed by prefix if given.
        """
        y_true = list(y_true)
        y_pred = list(y_pred)
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 and len(set(y_pred)) > 1 else 0.0
        bleu = self.compute_bleu(y_true, y_pred)
        p_micro = precision_score(y_true, y_pred, average=None, zero_division=0)
        r_micro = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        if p_micro.size < 3:
            p_micro = np.resize(p_micro, 3)
            r_micro = np.resize(r_micro, 3)
            f1_per = np.resize(f1_per, 3)
        out = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "mcc": mcc,
            "bleu": bleu,
            "f1_negative": float(f1_per[0]),
            "f1_neutral": float(f1_per[1]),
            "f1_positive": float(f1_per[2]),
            "precision_negative": float(p_micro[0]),
            "precision_neutral": float(p_micro[1]),
            "precision_positive": float(p_micro[2]),
            "recall_negative": float(r_micro[0]),
            "recall_neutral": float(r_micro[1]),
            "recall_positive": float(r_micro[2]),
            "confusion_matrix": cm,
        }
        if prefix:
            out = {f"{prefix}{k}": v for k, v in out.items()}
        return out

    def compute_bleu(self, y_true: list[int], y_pred: list[int]) -> float:
        """
        Compute corpus BLEU on label sequences.
        Each label (0/1/2) treated as a single-token 'sentence'.
        Uses smoothing to handle zero-count n-grams.
        Returns float 0-1. Included to satisfy project requirements.
        """
        if corpus_bleu is None or SmoothingFunction is None:
            return 0.0
        references = [[[str(t)]] for t in y_true]
        hypotheses = [str(p) for p in y_pred]
        smooth = SmoothingFunction()
        try:
            score = corpus_bleu(references, hypotheses, smoothing_function=smooth.method1)
        except Exception:
            score = 0.0
        return min(1.0, max(0.0, score))

    def plot_confusion_matrix(
        self,
        y_true: list[int] | np.ndarray,
        y_pred: list[int] | np.ndarray,
        save_path: Path | str,
        title: str = "",
    ) -> None:
        """Seaborn heatmap, normalized by row, saved as PNG."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        cm = confusion_matrix(y_true, y_pred)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sum > 0, cm / row_sum, 0)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm_norm,
            xticklabels=self.LABEL_NAMES,
            yticklabels=self.LABEL_NAMES,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            ax=ax,
            vmin=0,
            vmax=1,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        if title:
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def plot_training_curves(
        self,
        history: dict[str, Any],
        save_path: Path | str,
    ) -> None:
        """Two-panel: loss curves + val F1 curve."""
        hist = history.get("history", history)
        if not hist:
            return
        epochs = [h["epoch"] for h in hist if h.get("split") == "train"]
        train_loss = [h.get("loss", h.get("train_loss", 0)) for h in hist if h.get("split") == "train"]
        val_loss = [h.get("loss", h.get("val_loss", 0)) for h in hist if h.get("split") == "val"]
        val_f1 = [h.get("val_macro_f1", h.get("macro_f1", 0)) for h in hist if h.get("split") == "val"]
        if not epochs:
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        if train_loss:
            ax1.plot(epochs[: len(train_loss)], train_loss, label="Train loss")
        if val_loss:
            ax1.plot(epochs[: len(val_loss)], val_loss, label="Val loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.set_title("Loss")
        if val_f1:
            ax2.plot(epochs[: len(val_f1)], val_f1, label="Val Macro-F1")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Macro-F1")
        ax2.legend()
        ax2.set_title("Validation Macro-F1")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def build_comparison_table(self, results_list: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Cross-experiment comparison table.
        Columns: Model | Strategy | Framework | Accuracy |
                 Macro-F1 | BLEU | F1-Neg | F1-Neu | F1-Pos |
                 Trainable Params | Train Time (min)
        """
        rows = []
        for r in results_list:
            rows.append({
                "Model": r.get("model_name", ""),
                "Strategy": r.get("strategy", ""),
                "Framework": r.get("framework", ""),
                "Accuracy": r.get("test_accuracy"),
                "Macro-F1": r.get("test_macro_f1"),
                "BLEU": r.get("test_bleu"),
                "F1-Neg": r.get("test_f1_negative"),
                "F1-Neu": r.get("test_f1_neutral"),
                "F1-Pos": r.get("test_f1_positive"),
                "Trainable Params": r.get("trainable_params"),
                "Train Time (min)": r.get("training_time_minutes"),
            })
        return pd.DataFrame(rows)
