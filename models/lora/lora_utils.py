"""
LoRA utility functions: parameter counts and comparison table.
"""

from typing import Any

import pandas as pd
import torch.nn as nn

from models.lora.lora_layer import LoRALinear


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total, trainable, and frozen parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def get_lora_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return list of (name, module) for all LoRALinear layers."""
    result: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            result.append((name, module))
    return result


def compare_parameter_counts(
    full_stats: dict[str, Any],
    lora_stats: dict[str, Any],
) -> pd.DataFrame:
    """
    Returns comparison DataFrame:
      | Metric | Full Fine-Tuning | LoRA |
      | Total params | X | X |
      | Trainable params | X | X |
      | Trainable % | 100% | ~1-3% |
      | Reduction factor | 1x | ~48x |
    """
    full_total = full_stats.get("total_parameters", full_stats.get("total", 0))
    full_train = full_stats.get("trainable_parameters", full_stats.get("trainable", full_total))
    lora_total = lora_stats.get("total_parameters", lora_stats.get("total", 0))
    lora_train = lora_stats.get("trainable_parameters", lora_stats.get("trainable", 0))
    full_pct = (100.0 * full_train / full_total) if full_total else 0
    lora_pct = (100.0 * lora_train / lora_total) if lora_total else 0
    full_red = (full_total / full_train) if full_train else 0
    lora_red = (lora_total / lora_train) if lora_train else 0
    return pd.DataFrame(
        {
            "Metric": [
                "Total params",
                "Trainable params",
                "Trainable %",
                "Reduction factor",
            ],
            "Full Fine-Tuning": [
                full_total,
                full_train,
                f"{full_pct:.1f}%",
                f"{full_red:.1f}x",
            ],
            "LoRA": [
                lora_total,
                lora_train,
                f"{lora_pct:.1f}%",
                f"{lora_red:.1f}x",
            ],
        }
    )
