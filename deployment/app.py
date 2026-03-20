#FASTAPI INFERENCE SERVER 

import time
import re
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api_model_loader import load_model

#globals set at startup
model = None
tokenizer = None
model_name = "not_loaded"
device = "cpu"
start_time = None
total_preds = 0

id2label = {0: "negative", 1: "neutral", 2: "positive"}
arabic_re = re.compile(r"[\u0600-\u06FF]")
latin_re = re.compile(r"[a-zA-Z]")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, model_name, device, start_time
    start_time = time.perf_counter()
    model, tokenizer, model_name, device = load_model()
    yield

app = FastAPI(
    title="Arabic-English Sentiment Analysis API",
    description=(
        "Gulf Arabic-English Code-Switched Sentiment Analysis\n\n"
        "**Model:** MARBERTv2 Fine-Tuned on Gulf Arabic social media\n\n"
        "**Classes:** negative (0) / neutral (1) / positive (2)\n\n"
        "**Use the `/predict` endpoint below to test the model.**\n"
        "Enter any Arabic, English, or mixed Arabic-English text."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

#Request and response schemas
class PredictRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "الخدمة كانت really bad وايد disappointed"
            }
        }

class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict
    arabic_ratio: float
    text_type: str
    model_used: str
    inference_ms: float

class BatchPredictRequest(BaseModel):
    texts: list[str]

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "الخدمة كانت really bad وايد disappointed",
                    "وايد حلو التطبيق best experience ever",
                    "التحديث الجديد صار ننتظر نشوف",
                ]
            }
        }


#Helper functions

def _text_type(text): #Classifies whether text is code-switched, pure Arabic, or pure English
    has_ar = bool(arabic_re.search(text))
    has_en = bool(latin_re.search(text))
    if has_ar and has_en:
      return "code_switched"
    if has_ar:
      return "pure_arabic"
    if has_en:
      return "pure_english"
    return "other"

def _arabic_ratio(text): #Filtering out characters that are Arabic script.
    if not text: return 0.0
    ar = len(arabic_re.findall(text))
    return round(ar / len(text), 3)

def _run_inference(text):
    global total_preds
    t0 = time.perf_counter()
    enc = tokenizer(text, padding=True, truncation=True,
                    max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().tolist()
    pred_id = int(torch.tensor(probs).argmax().item())
    total_preds += 1
    return {
        "sentiment": id2label[pred_id],
        "confidence": round(probs[pred_id], 4),
        "probabilities": {id2label[i]: round(p, 4) for i, p in enumerate(probs)},
        "arabic_ratio": _arabic_ratio(text),
        "text_type": _text_type(text),
        "model_used": model_name,
        "inference_ms":  round((time.perf_counter() - t0) * 1000, 2),
    }

#Routes

@app.get("/", summary="API info")
async def root():
    """Returns API description and available endpoints."""
    return {
        "name": "Arabic-English Sentiment Analysis API",
        "description": "Gulf Arabic-English Code-Switched Sentiment Analysis",
        "model": model_name,
        "endpoints": {
            "POST /predict": "Single text prediction",
            "POST /batch_predict": "Batch prediction (up to 32 texts)",
            "GET /health": "Service status",
            "GET /examples": "Sample Gulf Arabic-English inputs",
            "GET /docs": "Interactive Swagger UI",
        }
    }

@app.post("/predict", response_model=PredictResponse, summary="Predict sentiment")
async def predict(req: PredictRequest): #Predict sentiment for a single Arabic-English text and returns the predicted class (positive / negative / neutral), confidence score, per-class probabilities, text type, arabic character ratio, and inference time.
    if not req.text or not req.text.strip():
        raise HTTPException(422, "Text cannot be empty")
    if len(req.text) > 512:
        raise HTTPException(422, "Text too long — maximum 512 characters")
    if not (arabic_re.search(req.text) or latin_re.search(req.text)):
        raise HTTPException(422, "Text must contain Arabic or Latin characters")
    if model is None:
        raise HTTPException(503, "Model not loaded — check server startup logs")
    return PredictResponse(**_run_inference(req.text))


@app.post("/batch_predict", summary="Batch predict sentiment")
async def batch_predict(req: BatchPredictRequest): #Predict sentiment for a list of predefined texts
    if not req.texts:
        raise HTTPException(422, "Texts list cannot be empty")
    if len(req.texts) > 32:
        raise HTTPException(422, "Maximum 32 texts per batch request")
    if model is None:
        raise HTTPException(503, "Model not loaded — check server startup logs")
    t0 = time.perf_counter()
    results = [_run_inference(t) for t in req.texts]
    total = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "predictions": results,
        "batch_size": len(results),
        "total_ms": total,
        "avg_ms_per_sample": round(total / len(results), 2),
    }

@app.get("/health", summary="Service health check")
async def health(): #Returns current model name, device, uptime, and total predictions served
    return {
        "status": "healthy" if model else "model_not_loaded",
        "model": model_name,
        "device": str(device),
        "uptime_seconds": round(time.perf_counter() - start_time, 1) if start_time else 0,
        "total_predictions": total_preds,
    }

@app.get("/examples", summary="Sample Gulf Arabic-English inputs")
async def examples(): #Returns example inputs covering different text types
    return [
        {"text": "الخدمة كانت really bad وايد disappointed", "expected": "negative", "type": "code_switched"},
        {"text": "وايد حلو التطبيق، best experience ever", "expected": "positive", "type": "code_switched"},
        {"text": "التحديث الجديد صار الحين، ننتظر نشوف", "expected": "neutral", "type": "pure_arabic"},
        {"text": "التطبيق ما يشتغل الحين ليش؟", "expected": "negative", "type": "pure_arabic"},
        {"text": "يا رجل الـ delivery تاخر وايد يبغي يصلحون", "expected": "negative", "type": "code_switched"},
        {"text": "التطبيق ممتاز وايد سريع amazing app", "expected": "positive", "type": "code_switched"},
    ]