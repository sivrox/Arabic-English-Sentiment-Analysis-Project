import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    checkpoint = Path("models/best-model/best_marbert_fft")

    print(f" Checking model path: {checkpoint}")
    print(f"Exists: {checkpoint.exists()}")

    if checkpoint.exists():
        print("Loading fine-tuned model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(checkpoint),
            num_labels=3
        ).to(dev)

        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
        model_name = "MARBERT Fine-Tuned"
    else:
        print("Loading base model (fallback)...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "UBC-NLP/MARBERTv2",
            num_labels=3
        ).to(dev)

        tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERTv2")

        model_name = "MARBERT Base"

    model.eval()

    return model, tokenizer, model_name, dev