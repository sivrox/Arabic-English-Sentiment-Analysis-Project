# Gulf Arabic–English Sentiment Analysis

Transfer learning pipeline for 3-class sentiment analysis (Positive / Negative / Neutral)
on Gulf Arabic–English code-switched social media text.

Group Project | CSCI316 — Big Data Mining and Applications | University of Wollongong in Dubai

---

## Model

**MARBERTv2** (UBC-NLP/MARBERTv2)
- Encoder-only transformer pretrained on 1 billion Arabic dialect tweets
- 163M parameters, 91,359 Arabic token IDs in vocabulary
- Two transfer strategies: Full Fine-Tuning and LoRA from Scratch

---

## Run Training

Open in Google Colab (T4 GPU/A100 GPU required):

1. `notebooks/pytorch_training_pipelinw.ipynb` — Raw PyTorch training loop, Full FT + LoRA from scratch
2. `notebooks/huggingface_training_pipeline.ipynb` — HuggingFace Trainer, Full FT + LoRA (peft library)
3. 
Both notebooks save results to `results/` and checkpoints to Google Drive.
Goggle Drive Link: https://drive.google.com/drive/folders/1DHzgJGJghjrOlE_yEFOC2RB1TXK0PO1A?usp=sharing
---

## Run Deployment
```bash
# Place fine-tuned checkpoint at deployment/models/best_model/best_marbert_fft/

cd deployment
docker build -t arabic-english-sentiment-analysis-api .
docker run -p 8001:8000 arabic-english-sentiment-analysis-api

# Open: http://localhost:8000/docs
```

API Endpoints:
- `POST /predict` — Single text prediction
- `POST /batch_predict` — Batch prediction (max 32 texts)
- `GET /health` — Service status and model info
- `GET /examples` — Sample Gulf Arabic–English inputs
- `GET /docs` — Interactive Swagger UI

---

## Project Structure
```
peft_implementation.py          #LoRA from scratch

preprocessing/
  data-scrapers                 #Online data scraping scrpits
  build_dataset.py              #Merges raw sources into unified_raw.csv
  preprocessor.py               #Full cleaning pipeline and DataLoader construction
  datasets/
    raw/                        #Source files
    lexicon/
    processed/
      unified_raw.csv           #213,255 rows combined
      cleaned_dataset.csv       #Preprocessed Dataset for training

notebooks/
  pytorch_training_pipeline.ipynb     #PyTorch pipeline — Full FT + LoRA Scratch
  huggingface_training_pipeline.ipynb #HuggingFace pipeline — Full FT + LoRA (peft)

evaluation/
  css_evaluation.ipynb          #CSS cultural metric evaluation

deployment/
  app.py                        #FastAPI inference server
  api_model_loader.py           #Model loading with checkpoint fallback
  Dockerfile
  docker-compose.yml
  requirements.txt

results/
  pytorch/                      #PyTorch training results and plots
  huggingface/                  #HuggingFace training results and plots
  cross_framework_comparison.csv
  css_results.json
```

---

## Dataset

Built from 8 sources — no single Gulf Arabic–English sentiment dataset exists:

| Source | Type | Labeling |
|---|---|---|
| arbml Arabic Sentiment Twitter Corpus | Pure Arabic | Manual |
| ASTD Arabic Sentiment Tweets | Pure Arabic | Manual |
| Company Reviews (Kaggle) | Mixed | Star ratings |
| MagedSaeed CS Text (HuggingFace) | Code-switched | AraSenti lexicon |
| ArE-CSTD (SDAIA/Kaggle) | Synthetic code-switched | AraSenti lexicon |
| App store reviews (UAE) | Mixed | Star ratings |
| YouTube comments (Gulf) | Mixed | Weak labels (gold test only) |

**213,255 total samples** | 76.5% code-switched | 23.5% pure Arabic

---

## CSS — Code-Switch Sensitivity Score

A custom cultural metric measuring whether the model genuinely reads Arabic tokens
or relies on English keywords. Computed by masking Arabic tokens and measuring
the drop in model confidence.

- CSS > 0.25 → Strong Arabic engagement
- CSS 0.10–0.25 → Moderate
- CSS < 0.10 → Weak (English-dependent)

**MARBERT Full FT CSS: 0.6925** (355 qualifying code-switched test samples)

---

## Requirements

Training: Google Colab Pro, T4 GPU/A100 GPU, PyTorch 2.2.0, Transformers 4.41.0, peft 0.10.0

Deployment: Docker, Python 3.11, FastAPI, Uvicorn

---

## References

1. Abdul-Mageed et al. (2021). ARBERT and MARBERT. ACL 2021.
2. Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
3. Al-Twairesh et al. (2016). AraSenTi. ACL 2016.
4. SDAIA (2024). ArE-CSTD. Kaggle.
