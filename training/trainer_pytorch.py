"""
Raw PyTorch training loop for SentimentGulf.
Handles full fine-tuning and LoRA for any encoder model.
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.standard_metrics import SentimentEvaluator
from training.callbacks import EarlyStopping, MetricsLogger
from training.optimizers import get_differential_optimizer, get_linear_warmup_scheduler

log = logging.getLogger(__name__)


class PyTorchSentimentTrainer:
    """
    Complete raw PyTorch training loop.
    Handles full fine-tuning and LoRA for any encoder model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: dict,
        experiment_name: str,
        output_dir: Path | str,
        device: Optional[torch.device] = None,
        model_name: str = "model",
        strategy: str = "full_finetuning",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model_name = model_name
        self.strategy = strategy
        train_cfg = config.get("training", {})
        self.max_epochs = train_cfg.get("max_epochs", 10)
        self.gradient_clip = train_cfg.get("gradient_clip_max_norm", 1.0)
        self.primary_metric = config.get("evaluation", {}).get("primary_metric", "macro_f1")
        lr = train_cfg.get("lora", {}).get("learning_rate") if strategy == "lora" else train_cfg.get("full_finetuning", {}).get("learning_rate", 2e-5)
        self.optimizer = get_differential_optimizer(
            model,
            base_lr=lr,
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )
        num_steps = len(train_loader) * self.max_epochs
        self.scheduler = get_linear_warmup_scheduler(
            self.optimizer,
            num_training_steps=num_steps,
            warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        )
        ckpt_path = self.output_dir / "best_checkpoint.pt"
        self.early_stopping = EarlyStopping(
            patience=train_cfg.get("early_stopping_patience", 3),
            metric=self.primary_metric,
            mode="max",
            checkpoint_path=ckpt_path,
        )
        self.metrics_logger = MetricsLogger(self.output_dir / "history.json")
        self.evaluator = SentimentEvaluator()
        self.history: list[dict] = []

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch["label"].to(self.device)
            self.optimizer.zero_grad()
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out["loss"]
            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        return {"loss": avg_loss}

    def evaluate(self, dataloader: DataLoader, split_name: str) -> dict[str, Any]:
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probs: list[list[float]] = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch["label"].to(self.device)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out["logits"]
                preds = logits.argmax(dim=-1).cpu().tolist()
                probs = torch.softmax(logits, dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())
                all_probs.extend(probs)
        return self.evaluator.compute_all_metrics(all_labels, all_preds, all_probs, prefix=f"{split_name}_")

    def train(self) -> dict[str, Any]:
        start_wall = time.perf_counter()
        for epoch in range(self.max_epochs):
            train_metrics = self.train_one_epoch(epoch)
            self.metrics_logger.log(epoch, "train", train_metrics)
            val_metrics = self.evaluate(self.val_loader, "val")
            self.metrics_logger.log(epoch, "val", val_metrics)
            primary_val = val_metrics.get(f"val_{self.primary_metric}", val_metrics.get(self.primary_metric, 0.0))
            stop = self.early_stopping.step(primary_val, self.model, epoch)
            self.history.append({"epoch": epoch, "split": "train", **train_metrics})
            self.history.append({"epoch": epoch, "split": "val", **val_metrics})
            log.info("Epoch %d train_loss=%.4f val_%s=%.4f", epoch, train_metrics["loss"], self.primary_metric, primary_val)
            if stop:
                break
        self.training_time_minutes = (time.perf_counter() - start_wall) / 60.0
        self.metrics_logger.save()
        self.model = self.early_stopping.load_best_model(self.model)
        return {"history": self.history}

    def run_final_evaluation(self) -> dict[str, Any]:
        """
        Returns standardized results dict with keys:
        model_name, strategy, framework, test_accuracy,
        test_macro_f1, test_bleu, test_f1_negative,
        test_f1_neutral, test_f1_positive,
        trainable_params, training_time_minutes,
        inference_time_ms_per_sample
        """
        self.model.eval()
        test_metrics = self.evaluate(self.test_loader, "test")
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # Inference timing: 100 forward passes
        n_time = 100
        start = time.perf_counter()
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                n_time -= input_ids.shape[0]
                if n_time <= 0:
                    break
        elapsed = time.perf_counter() - start
        batch_count = 100 - n_time if n_time < 100 else sum(batch["input_ids"].shape[0] for batch in self.test_loader)
        inference_ms = (elapsed * 1000.0 / max(1, batch_count)) if batch_count else 0.0
        training_minutes = getattr(self, "training_time_minutes", 0.0)
        return {
            "model_name": self.model_name,
            "strategy": self.strategy,
            "framework": "pytorch",
            "test_accuracy": test_metrics.get("test_accuracy", test_metrics.get("accuracy", 0)),
            "test_macro_f1": test_metrics.get("test_macro_f1", test_metrics.get("macro_f1", 0)),
            "test_bleu": test_metrics.get("test_bleu", test_metrics.get("bleu", 0)),
            "test_f1_negative": test_metrics.get("test_f1_negative", test_metrics.get("f1_negative", 0)),
            "test_f1_neutral": test_metrics.get("test_f1_neutral", test_metrics.get("f1_neutral", 0)),
            "test_f1_positive": test_metrics.get("test_f1_positive", test_metrics.get("f1_positive", 0)),
            "trainable_params": trainable_params,
            "training_time_minutes": training_minutes,
            "inference_time_ms_per_sample": inference_ms,
        }
