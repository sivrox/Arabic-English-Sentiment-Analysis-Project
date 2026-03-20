#IMPLEMENTATION OF LORA FROM SCRATCH

import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear layer and adds a trainable LoRA update path.
    The original weights are never changed — only lora_a and lora_b train.
    """

    def __init__(self, linear, r=8, alpha=16.0, dropout=0.1):
        super().__init__()
        self.r = r
        self.scaling = alpha / r  # applied to the LoRA output before adding to base
        self.linear  = linear

        # Freeze the original pretrained weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        d_out, d_in = linear.weight.shape

        # lora_a is initialized randomly, lora_b is zeros
        # zeros for lora_b means the update starts at zero (no change at init)
        self.lora_a = nn.Parameter(torch.randn(r, d_in) * 0.02)
        self.lora_b = nn.Parameter(torch.zeros(d_out, r))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # base output from frozen weights + scaled low-rank update
        return self.linear(x) + (self.dropout(x) @ self.lora_a.T @ self.lora_b.T) * self.scaling


def inject_lora(model, target_modules=("query", "value"), r=8, alpha=16.0, dropout=0.1):
    """
    Freezes all model weights, then replaces target Linear layers with LoRALinear.

    After calling this, only the lora_a and lora_b matrices in the replaced
    layers are trainable. Everything else is frozen.

    Note: the classifier head is also frozen by this function.
    Unfreeze it manually in the notebook after calling inject_lora().

    Args:
        model          : a HuggingFace transformer model
        target_modules : which layer name endings to replace with LoRA
        r, alpha, dropout : passed to LoRALinear

    Returns:
        the modified model (edited in place)
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    replaced = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # Get the parent module so we can replace the child attribute
                parts  = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
                replaced += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied: {replaced} layers replaced")
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


def count_parameters(model):
    """Returns trainable and total parameter counts as a dict."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "pct_trainable": round(100 * trainable / total, 2),
    }


#Verify LoRA Implementation

if __name__ == "__main__":
    print("\nTesting LoRA implementation...\n")

    # Small test model with the same structure as a transformer attention block
    class small_model(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)
            self.dense = nn.Linear(64, 64)

        def forward(self, x):
            return self.dense(self.query(x) + self.value(x))

    model = small_model()
    x = torch.randn(2, 10, 64)

    params_before = count_parameters(model)
    print(f"Before LoRA: {params_before['trainable']:,} trainable parameters")

    model = inject_lora(model, target_modules=("query", "value"), r=4, alpha=8.0)
    params_after = count_parameters(model)
    print(f"After LoRA:  {params_after['trainable']:,} trainable parameters")

    # Forward pass check
    out = model(x)
    print(f"\nForward pass output shape: {out.shape}  (expected torch.Size([2, 10, 64]))")

    # Gradient check — only LoRA params should receive gradients
    out.sum().backward()
    grads = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
    print(f"Parameters with gradients: {grads}\n")