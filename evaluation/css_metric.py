"""
Code-Switch Sensitivity Score (CSS) — Custom Cultural Metric.

Required by project spec: 'Beyond standard metrics, you must
propose and justify one culturally-aware evaluation metric.'

CSS measures whether the model genuinely relies on the Arabic
component of code-switched text, or whether it achieves good
performance by only reading English tokens.

Interpretation:
  CSS > 0.25: Strong Arabic engagement (desirable)
  CSS 0.10-0.25: Moderate engagement
  CSS < 0.10: Model largely ignoring Arabic
"""

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Arabic Unicode range (basic and extended)
ARABIC_RANGES = [
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
]


def _is_arabic_char(c: str) -> bool:
    if not c:
        return False
    code = ord(c[0])
    return any(lo <= code <= hi for lo, hi in ARABIC_RANGES)


class CSSEvaluator:
    def __init__(self, model: nn.Module, tokenizer: Any, device: str | torch.device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)
        self.model.eval()
        self._vocab_arabic: Optional[set[int]] = None

    def _ensure_vocab_arabic(self) -> set[int]:
        if self._vocab_arabic is not None:
            return self._vocab_arabic
        vocab = self.tokenizer.get_vocab()
        arabic_ids = set()
        for token, tid in vocab.items():
            if not token or token in ("<s>", "</s>", "<pad>", "<unk>"):
                continue
            if any(_is_arabic_char(c) for c in token):
                arabic_ids.add(tid)
        self._vocab_arabic = arabic_ids
        return arabic_ids

    def is_arabic_token(self, token_id: int) -> bool:
        return token_id in self._ensure_vocab_arabic()

    def mask_arabic_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pad_id: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Zero out (mask) Arabic token positions in input_ids so model
        cannot use them. Returns (masked_input_ids, attention_mask).
        """
        if pad_id is None:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        masked = input_ids.clone()
        arabic = self._ensure_vocab_arabic()
        for i in range(masked.shape[0]):
            for j in range(masked.shape[1]):
                if attention_mask[i, j].item() == 0:
                    continue
                if masked[i, j].item() in arabic:
                    masked[i, j] = pad_id
        return masked, attention_mask

    def get_confidence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[int, float]:
        """
        Run forward with full input and with Arabic masked.
        Returns (predicted_class, confidence_drop).
        confidence_drop = prob_full - prob_masked for the predicted class.
        """
        self.model.eval()
        with torch.no_grad():
            full_out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits_full = full_out.get("logits", full_out)
            if hasattr(logits_full, "logits"):
                logits_full = full_out.logits
            probs_full = torch.softmax(logits_full, dim=-1)
            pred_class = int(probs_full.argmax(dim=-1)[0].item())
            conf_full = probs_full[0, pred_class].item()
            masked_ids, masked_attn = self.mask_arabic_tokens(input_ids, attention_mask)
            masked_out = self.model(input_ids=masked_ids, attention_mask=masked_attn)
            logits_masked = masked_out.get("logits", masked_out)
            if hasattr(logits_masked, "logits"):
                logits_masked = masked_out.logits
            probs_masked = torch.softmax(logits_masked, dim=-1)
            conf_masked = probs_masked[0, pred_class].item()
            delta = conf_full - conf_masked
        return pred_class, max(0.0, delta)

    def compute_css(
        self,
        texts: list[str],
        true_labels: list[int],
        n_samples: int = 150,
        confidence_threshold: float = 0.80,
    ) -> dict[str, Any]:
        """
        Returns:
          css_score, interpretation, n_used,
          n_skipped_no_arabic, n_skipped_wrong_prediction,
          per_sample_df, class_breakdown
        """
        from torch.utils.data import DataLoader
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)
        else:
            attention_mask = attention_mask.to(self.device)
        true_labels = list(true_labels)
        n_skipped_no_arabic = 0
        n_skipped_wrong = 0
        deltas = []
        per_row = []
        arabic_ids = self._ensure_vocab_arabic()
        for i in range(min(n_samples, len(texts))):
            ids = input_ids[i : i + 1]
            attn = attention_mask[i : i + 1]
            has_ar = any(ids[0, j].item() in arabic_ids for j in range(ids.shape[1]) if attn[0, j].item() == 1)
            if not has_ar:
                n_skipped_no_arabic += 1
                continue
            pred_cls, delta = self.get_confidence(ids, attn)
            true_cls = true_labels[i] if i < len(true_labels) else -1
            if true_cls >= 0 and pred_cls != true_cls:
                n_skipped_wrong += 1
                continue
            probs_full = None
            with torch.no_grad():
                out = self.model(input_ids=ids, attention_mask=attn)
                logits = out.get("logits", out)
                if hasattr(logits, "logits"):
                    logits = out.logits
                probs = torch.softmax(logits, dim=-1)[0]
            conf = probs[pred_cls].item()
            if conf < confidence_threshold:
                continue
            deltas.append(delta)
            per_row.append({
                "index": i,
                "text_preview": texts[i][:50] if i < len(texts) else "",
                "true_label": true_cls,
                "pred_label": pred_cls,
                "confidence": conf,
                "confidence_drop": delta,
            })
        n_used = len(deltas)
        css_score = float(np.mean(deltas)) if deltas else 0.0
        if css_score > 0.25:
            interpretation = "Strong Arabic engagement (desirable)"
        elif css_score >= 0.10:
            interpretation = "Moderate engagement"
        else:
            interpretation = "Model largely ignoring Arabic"
        per_sample_df = pd.DataFrame(per_row)
        class_breakdown = per_sample_df.groupby("true_label")["confidence_drop"].agg(["mean", "count"]).to_dict() if not per_sample_df.empty else {}
        return {
            "css_score": css_score,
            "interpretation": interpretation,
            "n_used": n_used,
            "n_skipped_no_arabic": n_skipped_no_arabic,
            "n_skipped_wrong_prediction": n_skipped_wrong,
            "per_sample_df": per_sample_df,
            "class_breakdown": class_breakdown,
        }

    def plot_css_distribution(
        self,
        per_sample_df: pd.DataFrame,
        save_path: Path | str,
    ) -> None:
        """Violin or box plot of confidence_drop distribution."""
        if per_sample_df.empty or "confidence_drop" not in per_sample_df.columns:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.violinplot([per_sample_df["confidence_drop"].dropna()], positions=[0], showmeans=True)
        ax.set_ylabel("Confidence drop (Arabic masked)")
        ax.set_title("CSS: Distribution of reliance on Arabic tokens")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
