"""
LoRA (Low-Rank Adaptation) from scratch — no peft/transformers LoRA.

- LoRALayer: single low-rank delta
- LoRALinear: drop-in replacement for nn.Linear with LoRA
- apply_lora_to_transformer: wrap encoder with LoRA
- verify_lora_correctness: sanity checks
"""

from models.lora.lora_layer import LoRALayer, LoRALinear
from models.lora.lora_model import apply_lora_to_transformer, verify_lora_correctness
from models.lora.lora_utils import count_parameters, get_lora_layers, compare_parameter_counts

__all__ = [
    "LoRALayer",
    "LoRALinear",
    "apply_lora_to_transformer",
    "verify_lora_correctness",
    "count_parameters",
    "get_lora_layers",
    "compare_parameter_counts",
]
