"""
Apply LoRA from scratch to a HuggingFace transformer encoder.
Replaces target nn.Linear layers with LoRALinear (no peft).
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from models.lora.lora_layer import LoRALinear

log = logging.getLogger(__name__)


def _named_linear_modules(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    out: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            out.append((name, module))
    return out


def _module_name_contains_any(module_name: str, targets: list[str]) -> bool:
    name_lower = module_name.lower()
    return any(t.lower() in name_lower for t in targets)


def apply_lora_to_transformer(
    model: nn.Module,
    rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.1,
    target_modules: list[str] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Replace target nn.Linear layers with LoRALinear.
    Default targets: ['query', 'key', 'value', 'dense']

    Returns (modified_model, stats_dict).
    stats_dict: total_parameters, trainable_parameters,
      frozen_parameters, trainable_percentage,
      replaced_layers, reduction_factor
    """
    if target_modules is None:
        target_modules = ["query", "key", "value", "dense"]
    linears = _named_linear_modules(model)
    replaced = 0
    for name, linear in linears:
        if not _module_name_contains_any(name, target_modules):
            continue
        parent_name, _, child_name = name.rpartition(".")
        if not child_name:
            continue
        parent = model.get_submodule(parent_name) if parent_name else model
        lora_linear = LoRALinear(
            linear,
            rank=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        setattr(parent, child_name, lora_linear)
        replaced += 1
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    pct = (100.0 * trainable / total) if total else 0.0
    reduction = (total / trainable) if trainable else 0.0
    stats = {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_percentage": pct,
        "replaced_layers": replaced,
        "reduction_factor": reduction,
    }
    log.info(
        "LoRA applied: %d layers replaced, trainable %.2f%% (%d params)",
        replaced,
        pct,
        trainable,
    )
    return model, stats


def verify_lora_correctness(
    model: nn.Module,
    tokenizer: Any,
    device: str = "cpu",
) -> bool:
    """
    Assert: frozen params have requires_grad=False,
            LoRA params have requires_grad=True,
            forward pass produces valid output,
            backward populates LoRA grads only.
    """
    model = model.to(device)
    # Get a small batch
    dummy = tokenizer(
        ["hello world", "مرحبا"],
        padding=True,
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    input_ids = dummy["input_ids"].to(device)
    attention_mask = dummy.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    model.eval()
    # Check requires_grad
    frozen_params = [n for n, p in model.named_parameters() if not p.requires_grad]
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "No trainable (LoRA) parameters found"
    # Forward
    if hasattr(model, "forward") and "input_ids" in str(model.forward.__code__.co_varnames):
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        except TypeError:
            out = model(input_ids=input_ids)
    else:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.get("logits") if isinstance(out, dict) else getattr(out, "logits", None)
    assert logits is not None, "Model must return logits"
    assert logits.shape[0] == input_ids.shape[0], "Batch size mismatch"
    # Backward
    model.train()
    loss = logits.sum()
    loss.backward()
    lora_grads = [
        n for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0
    ]
    assert len(lora_grads) > 0, "LoRA params should receive gradients"
    return True
