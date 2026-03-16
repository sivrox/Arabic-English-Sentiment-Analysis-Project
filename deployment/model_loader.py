"""
Load sentiment model and tokenizer once for FastAPI app.
Used in lifespan so model is never loaded inside endpoint handlers.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

log = logging.getLogger(__name__)

# Default model name if no checkpoint found
DEFAULT_MODEL_NAME = "xlm-roberta-base"


def load_config(config_path: Optional[Path] = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def load_model_and_tokenizer(
    checkpoint_path: Optional[Path] = None,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> tuple[Any, Any, str]:
    """
    Load model and tokenizer. Prefer checkpoint if it exists and is valid.
    Returns (model, tokenizer, model_used_name).
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    tokenizer = None
    model = None
    used_name = model_name or DEFAULT_MODEL_NAME
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
            model = AutoModelForSequenceClassification.from_pretrained(
                str(checkpoint_path),
                num_labels=3,
            )
            model.to(dev)
            model.eval()
            used_name = str(checkpoint_path)
            log.info("Loaded model from checkpoint %s", checkpoint_path)
        except Exception as e:
            log.warning("Checkpoint load failed: %s; using HF model", e)
    if tokenizer is None or model is None:
        name = model_name or DEFAULT_MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(
            name,
            num_labels=3,
        )
        model.to(dev)
        model.eval()
        used_name = name
        log.info("Loaded model from HuggingFace: %s", name)
    return model, tokenizer, used_name
