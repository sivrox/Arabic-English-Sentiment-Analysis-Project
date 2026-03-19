from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class SentimentClassifier(nn.Module):
    """
    Sentiment classifier wrapping any HuggingFace encoder.
    Works with XLM-RoBERTa and MARBERT without modification.
    Architecture: Encoder -> [CLS] token -> Dropout -> Linear(3)
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = 3,
        dropout: float = 0.3,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = getattr(config, "hidden_size", 768)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        if freeze_base:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """
        Returns dict with "logits" and "loss" (if labels provided).
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use [CLS] or last_hidden_state[:, 0]
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return {"logits": logits, "loss": loss}

    def get_hidden_size(self) -> int:
        return self.classifier.in_features

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}
