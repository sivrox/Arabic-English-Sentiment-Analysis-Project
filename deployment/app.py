"""
FastAPI inference server for SentimentGulf.
Serves a browser-based interface for testing Gulf Arabic-English sentiment.
"""

import time
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from api_model_loader import load_model

# ── Globals ───────────────────────────────────────────────────────────────────
_model       = None
_tokenizer   = None
_model_name  = "not_loaded"
_device      = "cpu"
_start_time  = None
_total_preds = 0

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE  = re.compile(r"[a-zA-Z]")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _model_name, _device, _start_time
    _start_time = time.perf_counter()
    _model, _tokenizer, _model_name, _device = load_model()
    print(f"Model loaded: {_model_name} on {_device}")
    yield

app = FastAPI(
    title="Arabic Sentiment Analysis API",
    description="Gulf Arabic-English Code-Switched Sentiment Analysis",
    lifespan=lifespan,
)


# ── Pydantic models ────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict
    arabic_ratio: float
    text_type: str
    cleaned_text: str
    model_used: str
    inference_ms: float


# ── Helpers ────────────────────────────────────────────────────────────────────

def _text_type(text: str) -> str:
    has_ar = bool(ARABIC_RE.search(text))
    has_en = bool(LATIN_RE.search(text))
    if has_ar and has_en: return "code_switched"
    if has_ar:            return "pure_arabic"
    if has_en:            return "pure_english"
    return "other"

def _arabic_ratio(text: str) -> float:
    if not text: return 0.0
    ar = len(ARABIC_RE.findall(text))
    return round(ar / len(text), 3)

def _predict(text: str) -> dict:
    global _total_preds
    t0 = time.perf_counter()
    enc = _tokenizer(
        text, padding=True, truncation=True,
        max_length=128, return_tensors="pt"
    )
    inputs = {k: v.to(_device) for k, v in enc.items()}
    with torch.no_grad():
        out   = _model(**inputs)
        probs = torch.softmax(out.logits, dim=-1)[0].cpu().tolist()
    pred_id = int(torch.tensor(probs).argmax().item())
    _total_preds += 1
    return {
        "sentiment"     : ID2LABEL[pred_id],
        "confidence"    : round(probs[pred_id], 4),
        "probabilities" : {ID2LABEL[i]: round(p, 4) for i, p in enumerate(probs)},
        "arabic_ratio"  : _arabic_ratio(text),
        "text_type"     : _text_type(text),
        "cleaned_text"  : text,
        "model_used"    : _model_name,
        "inference_ms"  : round((time.perf_counter() - t0) * 1000, 2),
    }


# ── HTML Interface ─────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SentimentGulf — Arabic-English Sentiment Analysis</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f4f8; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 30px 16px; }
  .card { background: white; border-radius: 16px; padding: 36px; max-width: 760px; width: 100%; box-shadow: 0 4px 24px rgba(0,0,0,0.08); }
  h1 { color: #1a365d; font-size: 1.8rem; margin-bottom: 6px; }
  .subtitle { color: #718096; font-size: 0.95rem; margin-bottom: 28px; }
  label { font-weight: 600; color: #2d3748; display: block; margin-bottom: 8px; }
  textarea { width: 100%; border: 2px solid #e2e8f0; border-radius: 10px; padding: 14px; font-size: 1rem; resize: vertical; min-height: 120px; outline: none; direction: auto; transition: border 0.2s; }
  textarea:focus { border-color: #4299e1; }
  button { margin-top: 14px; width: 100%; padding: 14px; background: #2b6cb0; color: white; border: none; border-radius: 10px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: background 0.2s; }
  button:hover { background: #2c5282; }
  button:disabled { background: #a0aec0; cursor: not-allowed; }
  .examples { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
  .ex-btn { background: #ebf8ff; color: #2b6cb0; border: 1px solid #bee3f8; border-radius: 20px; padding: 6px 14px; font-size: 0.82rem; cursor: pointer; transition: all 0.2s; }
  .ex-btn:hover { background: #bee3f8; }
  #result { margin-top: 28px; display: none; }
  .sentiment-badge { display: inline-block; padding: 10px 24px; border-radius: 30px; font-size: 1.4rem; font-weight: 700; margin-bottom: 18px; }
  .positive  { background: #c6f6d5; color: #22543d; }
  .negative  { background: #fed7d7; color: #742a2a; }
  .neutral   { background: #e2e8f0; color: #2d3748; }
  .conf-bar-wrap { background: #e2e8f0; border-radius: 8px; height: 12px; margin-bottom: 6px; }
  .conf-bar { height: 12px; border-radius: 8px; transition: width 0.5s; }
  .meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 18px; }
  .meta-item { background: #f7fafc; border-radius: 8px; padding: 12px; }
  .meta-label { font-size: 0.75rem; color: #718096; margin-bottom: 2px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
  .meta-val { font-size: 0.95rem; color: #2d3748; font-weight: 500; }
  .prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
  .prob-label { width: 80px; font-size: 0.85rem; color: #4a5568; text-transform: capitalize; }
  .prob-num { width: 50px; text-align: right; font-size: 0.85rem; color: #2d3748; font-weight: 600; }
  .loading { color: #718096; font-style: italic; margin-top: 16px; }
  .error-msg { color: #c53030; background: #fff5f5; border: 1px solid #fed7d7; padding: 12px; border-radius: 8px; margin-top: 16px; }
  h3 { color: #2d3748; font-size: 1rem; margin-bottom: 12px; }
</style>
</head>
<body>
<div class="card">
  <h1>Arabic-English Sentiment Analyzer</h1>
  <p class="subtitle">Gulf Arabic–English Code-Switched Sentiment Analysis · MARBERT Full Fine-Tuning · CSCI316 Project 2</p>

  <label for="textInput">Enter Arabic-English text:</label>
  <textarea id="textInput" placeholder="Type or paste text here — Arabic, English, or mixed (e.g. الخدمة كانت really bad وايد disappointed)"></textarea>

  <div class="examples">
    <span style="font-size:0.8rem;color:#718096;align-self:center;">Try an example:</span>
    <button class="ex-btn" onclick="setExample('الخدمة كانت really bad وايد disappointed')">Negative CS</button>
    <button class="ex-btn" onclick="setExample('وايد حلو التطبيق، best experience ever الحين')">Positive CS</button>
    <button class="ex-btn" onclick="setExample('التحديث الجديد صار الحين، ننتظر نشوف')">Neutral Arabic</button>
    <button class="ex-btn" onclick="setExample('يا رجل الحين الـ delivery تاخر وايد، يبغي يصلحون')">Gulf + English</button>
    <button class="ex-btn" onclick="setExample('التطبيق ممتاز وايد سريع amazing app')">Positive Mixed</button>
  </div>

  <button id="predictBtn" onclick="predict()">Analyse Sentiment</button>

  <div id="result">
    <div id="sentBadge" class="sentiment-badge"></div>
    <h3>Class Probabilities</h3>
    <div id="probBars"></div>
    <div class="meta-grid" id="metaGrid"></div>
  </div>
  <div id="loadingMsg" class="loading" style="display:none">Analysing...</div>
  <div id="errorMsg" class="error-msg" style="display:none"></div>
</div>

<script>
function setExample(text) {
  document.getElementById('textInput').value = text;
}

async function predict() {
  const text = document.getElementById('textInput').value.trim();
  if (!text) { showError('Please enter some text.'); return; }

  document.getElementById('result').style.display = 'none';
  document.getElementById('errorMsg').style.display = 'none';
  document.getElementById('loadingMsg').style.display = 'block';
  document.getElementById('predictBtn').disabled = true;

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || 'Server error');
    }
    const data = await resp.json();
    renderResult(data);
  } catch (e) {
    showError(e.message);
  } finally {
    document.getElementById('loadingMsg').style.display = 'none';
    document.getElementById('predictBtn').disabled = false;
  }
}

function renderResult(d) {
  const emoji = {positive:'😊', negative:'😠', neutral:'😐'}[d.sentiment] || '';
  const badge = document.getElementById('sentBadge');
  badge.textContent = emoji + ' ' + d.sentiment.charAt(0).toUpperCase() + d.sentiment.slice(1);
  badge.className = 'sentiment-badge ' + d.sentiment;

  const colours = {positive:'#48bb78', negative:'#fc8181', neutral:'#a0aec0'};
  const probs = d.probabilities;
  document.getElementById('probBars').innerHTML = Object.entries(probs)
    .sort((a,b) => b[1]-a[1])
    .map(([label, prob]) => `
      <div class="prob-row">
        <span class="prob-label">${label}</span>
        <div class="conf-bar-wrap" style="flex:1">
          <div class="conf-bar" style="width:${(prob*100).toFixed(1)}%;background:${colours[label]||'#a0aec0'}"></div>
        </div>
        <span class="prob-num">${(prob*100).toFixed(1)}%</span>
      </div>`).join('');

  document.getElementById('metaGrid').innerHTML = `
    <div class="meta-item"><div class="meta-label">Confidence</div><div class="meta-val">${(d.confidence*100).toFixed(2)}%</div></div>
    <div class="meta-item"><div class="meta-label">Text Type</div><div class="meta-val">${d.text_type.replace('_',' ')}</div></div>
    <div class="meta-item"><div class="meta-label">Arabic Ratio</div><div class="meta-val">${(d.arabic_ratio*100).toFixed(1)}%</div></div>
    <div class="meta-item"><div class="meta-label">Inference Time</div><div class="meta-val">${d.inference_ms} ms</div></div>
    <div class="meta-item" style="grid-column:1/-1"><div class="meta-label">Model</div><div class="meta-val">${d.model_used}</div></div>
  `;

  document.getElementById('result').style.display = 'block';
}

function showError(msg) {
  const el = document.getElementById('errorMsg');
  el.textContent = 'Error: ' + msg;
  el.style.display = 'block';
}

document.getElementById('textInput').addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') predict();
});
</script>
</body>
</html>"""


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Browser interface for sentiment analysis."""
    return HTML_PAGE


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty")
    if len(req.text) > 512:
        raise HTTPException(status_code=422, detail="Text too long — max 512 characters")
    if not (ARABIC_RE.search(req.text) or LATIN_RE.search(req.text)):
        raise HTTPException(status_code=422, detail="Text must contain Arabic or English")
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return PredictResponse(**_predict(req.text))


@app.post("/batch_predict")
async def batch_predict(req: dict):
    texts = req.get("texts", [])
    if not texts or len(texts) > 32:
        raise HTTPException(status_code=422, detail="Provide 1–32 texts")
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    t0 = time.perf_counter()
    results = [_predict(t) for t in texts]
    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {"predictions": results, "total_ms": total_ms,
            "avg_ms": round(total_ms / len(results), 2)}


@app.get("/health")
async def health():
    return {
        "status" : "healthy" if _model else "model_not_loaded",
        "model" : _model_name,
        "device" : str(_device),
        "uptime_seconds": round(time.perf_counter() - _start_time, 1) if _start_time else 0,
        "total_predictions": _total_preds,
    }


@app.get("/examples")
async def examples():
    return [
        {"text": "الخدمة كانت really bad وايد disappointed", "expected": "negative", "note": "Code-switched negative"},
        {"text": "وايد حلو التطبيق، best experience ever", "expected": "positive", "note": "Code-switched positive"},
        {"text": "التطبيق ما يشتغل الحين ليش؟", "expected": "negative", "note": "Pure Gulf Arabic negative"},
        {"text": "التحديث الجديد صار، ننتظر نشوف", "expected": "neutral", "note": "Pure Arabic neutral"},
        {"text": "يا رجل الـ delivery تاخر وايد، يبغي يصلحون", "expected": "negative", "note": "Gulf dialect + English mixing"},
    ]