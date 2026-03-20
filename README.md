# Gulf Arabic-English Sentiment Analysis

Production-quality research project: **3-class sentiment analysis** (Positive / Negative / Neutral) for **Gulf Arabic–English code-switched** social media text.

Transfer learning for 3-class sentiment analysis on Gulf Arabic-English 
code-switched social media text.

## Models
- XLM-RoBERTa-base (Full Fine-Tuning + LoRA from Scratch)
- MARBERTv2 (Full Fine-Tuning + LoRA from Scratch)

## Results
| Model | Strategy | Test Macro-F1 |
|---|---|---|
| XLM-RoBERTa | Full FT | 0.8989 |
| XLM-RoBERTa | LoRA Scratch | 0.8912 |
| MARBERT | Full FT | **0.9122** |
| MARBERT | LoRA Scratch | 0.9094 |

CSS Score (MARBERT Full FT): **0.6604** (Strong Arabic engagement)

## Run Training
Open notebooks/ in Google Colab (T4 GPU required):
1. `01_pytorch_training.ipynb` — PyTorch raw training loop
2. `02_huggingface_training.ipynb` — HuggingFace Trainer

## Run Deployment
```bash
cd deployment
docker build -t arabic-english-sentiment-analysis-api .
docker run -p 8001:8000 -v "/path/to/models:/app/models" arabic-english-sentiment-analysis-api
# Open: http://localhost:8001/docs
```


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
