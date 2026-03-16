"""
Differential learning rate optimizer and linear warmup scheduler.
"""

from typing import Generator, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def _get_param_groups_by_depth(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
) -> list[dict]:
    """
    Build param groups with layer-wise LR scaling.
    Embeddings: base_lr * 0.1
    Layers 0-3: base_lr * 0.25
    Layers 4-7: base_lr * 0.5
    Layers 8-11: base_lr * 1.0
    Head: base_lr * 2.0
    """
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    groups: dict[str, dict] = {}
    default_lr = base_lr

    def lr_scale(name: str) -> float:
        name_lower = name.lower()
        if "embed" in name_lower:
            return 0.1
        if "encoder.layer" in name_lower or "layer." in name_lower:
            for i in range(12):
                if f".{i}." in name or f"layer.{i}." in name_lower:
                    if i <= 3:
                        return 0.25
                    if i <= 7:
                        return 0.5
                    return 1.0
        if "classifier" in name_lower or "pooler" in name_lower or "lm_head" in name_lower:
            return 2.0
        if "lora" in name_lower:
            return 1.0
        return 0.5

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        scale = lr_scale(name)
        lr = base_lr * scale
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        key = f"{lr}_{wd}"
        if key not in groups:
            groups[key] = {"params": [], "lr": lr, "weight_decay": wd}
        groups[key]["params"].append(p)
    return list(groups.values())


def get_differential_optimizer(
    model: nn.Module,
    base_lr: float,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """
    AdamW with differential learning rates by layer depth.
    Embeddings: base_lr * 0.1
    Layers 0-3: base_lr * 0.25
    Layers 4-7: base_lr * 0.5
    Layers 8-11: base_lr * 1.0
    Head: base_lr * 2.0
    LoRA-only params: filters out non-trainable automatically.
    """
    param_groups = _get_param_groups_by_depth(model, base_lr, weight_decay)
    if not param_groups:
        return AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    return AdamW(param_groups)


def get_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
) -> LambdaLR:
    """Linear warmup then linear decay to 0."""

    def lr_lambda(step: int) -> float:
        warmup_steps = int(num_training_steps * warmup_ratio)
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(num_training_steps - step) / float(max(1, num_training_steps - warmup_steps)))

    return LambdaLR(optimizer, lr_lambda)
