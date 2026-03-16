"""
LoRA (Low-Rank Adaptation) implemented from scratch.
Only torch and torch.nn — no peft, transformers, or any LoRA library.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation for a single linear projection.

    Mathematical formulation:
      Given frozen W_0 (out x in), LoRA parameterizes the update as:
          delta_W = (alpha / rank) * B @ A
      where A: (rank x in) initialized with kaiming_uniform
            B: (out x rank) initialized with zeros
      At init: B=0 so delta_W=0, preserving pretrained behavior.

    Forward returns ONLY the LoRA delta:
      lora_output = scaling * (dropout(x) @ A.T @ B.T)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.scaling = lora_alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns only the LoRA delta: scaling * dropout(x) @ A.T @ B.T
        """
        dropped = self.lora_dropout(x)
        # (batch, in) @ (in, rank) -> (batch, rank); then @ (rank, out) -> (batch, out)
        out = (dropped @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return out

    @property
    def trainable_params(self) -> int:
        return int(self.lora_A.numel() + self.lora_B.numel())


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with LoRA.
    Original linear completely frozen.
    forward = frozen_linear(x) + lora_layer(x)
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.linear = original_linear
        for p in self.linear.parameters():
            p.requires_grad = False
        in_f = original_linear.in_features
        out_f = original_linear.out_features
        self.lora = LoRALayer(
            in_features=in_f,
            out_features=out_f,
            rank=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA delta into weights permanently for inference."""
        W = self.linear.weight.data
        delta = (self.lora.lora_alpha / self.lora.rank) * (
            self.lora.lora_B.data @ self.lora.lora_A.data
        )
        merged = nn.Linear(
            self.linear.in_features,
            self.linear.out_features,
            bias=self.linear.bias is not None,
            device=W.device,
            dtype=W.dtype,
        )
        merged.weight.data = W + delta
        if self.linear.bias is not None:
            merged.bias.data = self.linear.bias.data.clone()
        return merged
