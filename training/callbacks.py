"""
Training callbacks: EarlyStopping and MetricsLogger.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class EarlyStopping:
    """
    Stop training when val metric stops improving.
    Auto-saves best checkpoint.
    """

    def __init__(
        self,
        patience: int = 3,
        metric: str = "macro_f1",
        mode: str = "max",
        checkpoint_path: Optional[Path] = None,
    ) -> None:
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.best_value: Optional[float] = None
        self.best_epoch: int = -1
        self.counter: int = 0
        self.best_state: Optional[dict[str, Any]] = None

    def step(
        self,
        current_value: float,
        model: nn.Module,
        epoch: int,
    ) -> bool:
        """
        Returns True if training should stop.
        """
        if self.best_value is None:
            self.best_value = current_value
            self.best_epoch = epoch
            self._save_best(model)
            return False
        if self.mode == "max":
            improved = current_value > self.best_value
        else:
            improved = current_value < self.best_value
        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
            self._save_best(model)
            return False
        self.counter += 1
        if self.counter >= self.patience:
            log.info("Early stopping at epoch %d (metric %s did not improve for %d epochs)", epoch, self.metric, self.patience)
            return True
        return False

    def _save_best(self, model: nn.Module) -> None:
        self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if self.checkpoint_path:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": self.best_state}, self.checkpoint_path)

    def load_best_model(self, model: nn.Module) -> nn.Module:
        device = next(model.parameters(), torch.tensor(0)).device
        if self.best_state is not None:
            state = {k: v.to(device) for k, v in self.best_state.items()}
            model.load_state_dict(state, strict=False)
        elif self.checkpoint_path and self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location=device)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)
        return model


class MetricsLogger:
    """Track and save training history to JSON."""

    def __init__(self, save_path: Path) -> None:
        self.save_path = Path(save_path)
        self.history: list[dict[str, Any]] = []

    def log(self, epoch: int, split: str, metrics: dict[str, Any]) -> None:
        entry = {"epoch": epoch, "split": split, **metrics}
        self.history.append(entry)

    def save(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_history(self) -> dict[str, Any]:
        return {"history": self.history}
