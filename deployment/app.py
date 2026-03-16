"""
FastAPI inference server for SentimentGulf.

Why FastAPI (per project spec: 'Flask/FastAPI'):
  - Async request handling for concurrent demo use
  - Auto-generated /docs endpoint for interactive testing
    (useful during live project demonstration)
  - Type validation via Pydantic (cleaner than Flask)
  - Better performance under load than Flask

Endpoints:
  POST /predict        Single text prediction
  POST /batch_predict  Batch prediction (max 32)
  GET  /health         Service status
  GET  /examples       5 hardcoded Gulf Arabic CS examples
  GET  /docs           Auto-generated API documentation (FastAPI built-in)
"""

import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add project root for imports
import sys
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from deployment.model_loader import load_config, load_model_and_tokenizer
from preprocessing.arabic_normalizer import GulfArabicNormalizer

# Arabic and Latin (English) Unicode ranges
ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
LATIN_RE = re.compile(r"[a-zA-Z]")


def _has_arabic_or_english(text: str) -> bool:
    return bool(ARABIC_RE.search(text) or LATIN_RE.search(text))


def _arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    ar = len(ARABIC_RE.findall(text))
    return ar / len(text) if text else 0.0


def _token_language(token: str) -> str:
    if ARABIC_RE.search(token):
        return "ar"
    if LATIN_RE.search(token):
        return "en"
    return "other"


# Global state set in lifespan
_model = None
_tokenizer = None
_model_name = "not_loaded"
_device = "cpu"
_start_time = None
_total_predictions = 0
_normalizer = GulfArabicNormalizer()
_MAX_TEXT_LENGTH = 512
_MAX_BATCH = 32


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _model_name, _device, _start_time
    _start_time = time.perf_counter()
    config = load_config()
    dep = config.get("deployment", {})
    best_path = dep.get("best_model_path", "models/checkpoints/best_model")
    path = _root / best_path if not Path(best_path).is_absolute() else Path(best_path)
    _model, _tokenizer, _model_name = load_model_and_tokenizer(
        checkpoint_path=path if path.exists() else None,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    _device = next(_model.parameters()).device
    yield
    # shutdown: nothing to clean


app = FastAPI(
    title="SentimentGulf API",
    description="3-class sentiment (positive/negative/neutral) for Gulf Arabic–English code-switched text",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(..., description="Input text to classify")


class TokenInfo(BaseModel):
    token: str
    language: str
    position: int


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    label_id: int
    probabilities: dict[str, float]
    token_analysis: list[TokenInfo]
    arabic_ratio: float
    cleaned_text: str
    model_used: str
    inference_time_ms: float


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., max_length=_MAX_BATCH)


def _validate_text(text: str) -> None:
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty")
    if len(text) > _MAX_TEXT_LENGTH:
        raise HTTPException(status_code=422, detail=f"Text too long, max {_MAX_TEXT_LENGTH} chars")
    if not _has_arabic_or_english(text):
        raise HTTPException(status_code=422, detail="Text must contain Arabic or English")


def _run_predict(text: str) -> dict[str, Any]:
    global _total_predictions
    cleaned = _normalizer.full_pipeline(text)
    enc = _tokenizer(
        cleaned,
        padding=True,
        truncation=True,
        max_length=_MAX_TEXT_LENGTH,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(_device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(_device)
    import torch as _torch
    t0 = time.perf_counter()
    with _torch.no_grad():
        out = _model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits if hasattr(out, "logits") else out["logits"]
    probs = _torch.softmax(logits, dim=-1)[0].cpu().tolist()
    inference_ms = (time.perf_counter() - t0) * 1000
    _total_predictions += 1
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    pred_id = int(logits.argmax(dim=-1)[0].item())
    token_analysis = []
    for i, tid in enumerate(input_ids[0].tolist()):
        tok = _tokenizer.decode([tid])
        if tok.strip():
            token_analysis.append(TokenInfo(token=tok.strip(), language=_token_language(tok), position=i))
    return {
        "sentiment": id2label[pred_id],
        "confidence": probs[pred_id],
        "label_id": pred_id,
        "probabilities": {id2label[i]: probs[i] for i in range(3)},
        "token_analysis": token_analysis,
        "arabic_ratio": _arabic_ratio(cleaned),
        "cleaned_text": cleaned,
        "model_used": _model_name,
        "inference_time_ms": inference_ms,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    _validate_text(req.text)
    result = _run_predict(req.text)
    return PredictResponse(**result)


@app.post("/batch_predict")
async def batch_predict(req: BatchPredictRequest) -> dict:
    if len(req.texts) > _MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"Batch size exceeds maximum {_MAX_BATCH}")
    for t in req.texts:
        _validate_text(t)
    t0 = time.perf_counter()
    results = [_run_predict(t) for t in req.texts]
    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "results": [PredictResponse(**r) for r in results],
        "batch_size": len(results),
        "total_time_ms": total_ms,
        "avg_time_ms_per_sample": total_ms / len(results) if results else 0,
    }


@app.get("/health")
async def health() -> dict:
    return {
        "model_name": _model_name,
        "device": str(_device),
        "is_loaded": _model is not None,
        "uptime_seconds": time.perf_counter() - _start_time if _start_time else 0,
        "total_predictions_served": _total_predictions,
    }


EXAMPLES = [
    {
        "id": 1,
        "description": "Pure Arabic negative complaint (app review)",
        "text": "التطبيق سيء جدا وما يشتغل الحين، خسارة وقت",
    },
    {
        "id": 2,
        "description": "Code-switched positive (Gulf markers + English)",
        "text": "وايد حلو الـ app، best experience الحين",
    },
    {
        "id": 3,
        "description": "Code-switched negative with emoji",
        "text": "ما نصحكم فيه، service زفت والفلوس ضاعت",
    },
    {
        "id": 4,
        "description": "Pure Arabic neutral observation",
        "text": "التحديث الجديد صار الحين، ننتظر ن شوف",
    },
    {
        "id": 5,
        "description": "Heavy CS with multiple Gulf dialect markers",
        "text": "يا رجل الحين الـ delivery تاخر وايد، يبغي يصلحون",
    },
]


@app.get("/examples")
async def examples() -> list[dict]:
    return EXAMPLES
