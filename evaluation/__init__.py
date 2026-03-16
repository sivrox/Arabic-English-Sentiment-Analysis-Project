"""
Evaluation for SentimentGulf.

- SentimentEvaluator: accuracy, macro-F1, BLEU, per-class F1, confusion matrix
- CSSEvaluator: Code-Switch Sensitivity (cultural metric)
"""

from evaluation.standard_metrics import SentimentEvaluator
from evaluation.css_metric import CSSEvaluator

__all__ = ["SentimentEvaluator", "CSSEvaluator"]
