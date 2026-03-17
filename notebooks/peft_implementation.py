"""
peft_implementation.py
----------------------
Standalone LoRA (Low-Rank Adaptation) implementation from scratch.
No external PEFT library required.

Reference: Hu et al. (2021) — "LoRA: Low-Rank Adaptation of Large Language Models"
           https://arxiv.org/abs/2106.09685

Key idea:
    For a frozen weight matrix W ∈ R^(d x k),
    LoRA adds a trainable low-rank update:
        ΔW = B · A    where B ∈ R^(d x r), A ∈ R^(r x k), r << d,k
    Forward: h = Wx + BAx * (alpha / r)

    This reduces trainable parameters from d*k → r*(d+k),
    e.g. for d=k=768, r=8: 589,824 → 12,288  (48x reduction)
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an nn.Linear layer with a trainable LoRA low-rank update.

    Args:
        linear   : pre-trained nn.Linear to wrap (weights are frozen)
        r        : rank of the decomposition (default 8)
        alpha    : scaling factor; effective lr = alpha/r * lr (default 16)
        dropout  : dropout rate applied to the LoRA path (default 0.1)
    """
    def __init__(self, linear: nn.Linear, r: int = 8,
                 alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.r       = r
        self.scaling = alpha / r
        self.linear  = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        d_out, d_in   = linear.weight.shape
        self.lora_A   = nn.Parameter(torch.randn(r, d_in) * 0.02)  # random init
        self.lora_B   = nn.Parameter(torch.zeros(d_out, r))         # zero init → ΔW=0 at t=0
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def extra_repr(self) -> str:
        d_out, d_in = self.linear.weight.shape
        return (f"in={d_in}, out={d_out}, r={self.r}, "
                f"scaling={self.scaling:.3f}, "
                f"params_saved={d_in*d_out - self.r*(d_in+d_out)}")


def inject_lora(model: nn.Module,
                target_modules: tuple = ("query", "value"),
                r: int = 8,
                alpha: float = 16.0,
                dropout: float = 0.1) -> nn.Module:
    """Freeze all model weights, then replace target Linear layers with LoRALinear.

    Args:
        model          : any nn.Module (typically a HuggingFace transformer)
        target_modules : tuple of layer name suffixes to replace (default q & v projections)
        r, alpha, dropout : passed through to LoRALinear

    Returns:
        The modified model (in-place) with only LoRA parameters trainable.
    """
    for param in model.parameters():
        param.requires_grad = False

    replaced = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parts  = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1],
                        LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
                replaced += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] Replaced {replaced} layers | "
          f"Trainable: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
    return model


def count_parameters(model: nn.Module) -> dict:
    """Utility: return trainable vs total parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable,
            "frozen": total - trainable, "pct_trainable": 100 * trainable / total}
