import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
 
#Loads model and tokenizer, moves model to available device
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
 
    checkpoint = Path("models/checkpoints/best_marbert_fft")
 
    print(f"Checking checkpoint: {checkpoint}")
    print(f"Exists: {checkpoint.exists()}")
 
    if checkpoint.exists():
        print("Loading fine-tuned MARBERT checkpoint...")
        model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint), num_labels=3).to(dev)
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
        model_name = "MARBERT Fine-Tuned"
    else:
        print("\nCheckpoint not found. Loading base MARBERT as fallback...")    #Fallback to base model if checkpoint is missing
        print("Warning: predictions will be unreliable without the fine-tuned checkpoint.")
        model = AutoModelForSequenceClassification.from_pretrained("UBC-NLP/MARBERTv2", num_labels=3).to(dev)
        tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERTv2")
        model_name = "MARBERT Base (not fine-tuned)"
 
    model.eval()
    print(f"Model ready: {model_name} on {device}")
    return model, tokenizer, model_name, dev