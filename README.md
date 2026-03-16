# Arabic-English Sentiment Analysis (SentimentGulf)

Production-quality research project: **3-class sentiment analysis** (Positive / Negative / Neutral) for **Gulf Arabic–English code-switched** social media text.

## Overview

- **Task**: 3-class sentiment classification  
- **Language**: Gulf Arabic–English code-switched text (UAE and broader Gulf)  
- **Models**: XLM-RoBERTa-base, MARBERTv2  
- **Transfer strategies**: Full fine-tuning and **LoRA from scratch** (raw PyTorch)  
- **Frameworks**: Raw PyTorch (manual loops) and HuggingFace Trainer  
- **Deployment**: FastAPI + Docker  

## Directory Structure

```
configs/config.yaml
preprocessing/
  build_dataset.py   (existing)
  arabic_normalizer.py
  back_translation.py
models/
  sentiment_classifier.py
  lora/              (LoRA from scratch, no PEFT)
training/
  dataset_loader.py
  trainer_pytorch.py
  callbacks.py
  optimizers.py
evaluation/
  standard_metrics.py   (Accuracy, Macro-F1, BLEU, per-class F1)
  css_metric.py         (Code-Switch Sensitivity — cultural metric)
notebooks/
  01_pytorch_training.ipynb
  02_huggingface_training.ipynb
deployment/
  app.py
  model_loader.py
  Dockerfile
  docker-compose.yml
  requirements.txt
```

## Setup

```bash
pip install -r deployment/requirements.txt
```

Data: place `master_dataset.csv` under `data/processed/` (or use `preprocessing/data/processed/` and set paths in config/notebooks).

## Training

- **Notebook 01**: Raw PyTorch training (full fine-tuning and LoRA from scratch in `models/lora/`).  
- **Notebook 02**: HuggingFace Trainer and PEFT LoRA.

Run from project root so that `preprocessing`, `models`, `training`, `evaluation` are importable.

## Evaluation

- **Primary metric**: Macro-F1.  
- **BLEU**: Included per project requirements (computed on label sequences for classification).  
- **CSS (Code-Switch Sensitivity)**: Custom cultural metric measuring reliance on Arabic tokens.

## API (FastAPI)

```bash
uvicorn deployment.app:app --reload --host 0.0.0.0 --port 8000
```

- `POST /predict` — single text  
- `POST /batch_predict` — batch (max 32)  
- `GET /health` — status  
- `GET /examples` — 5 Gulf Arabic CS examples  
- `GET /docs` — Swagger UI  

## Docker

```bash
docker-compose -f deployment/docker-compose.yml up --build
```

## Model Selection (Section 4 Methodology)

We use **XLM-RoBERTa-base** and **MARBERTv2** instead of the spec-suggested BLOOM, mT5/mBART, LaMini-FLAN-T5, AceGPT/Jais:

- **XLM-RoBERTa**: Strong cross-lingual encoder for Arabic–English code-switching.  
- **MARBERTv2**: Arabic-dialect model (1B Arabic tweets), better for Gulf dialect.  
- AceGPT/Jais are decoder-only; MARBERTv2 is better suited for encoder-based sentiment.

## License and Citation

Use for CSCI316 research project. Cite dataset and model sources as required by your course.
