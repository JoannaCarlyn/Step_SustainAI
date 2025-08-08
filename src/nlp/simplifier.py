import os
import torch
import pickle
import joblib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from textstat import flesch_reading_ease

MODEL_NAME = "google/flan-t5-small"
MODEL_DIR = "model/nlp_transformer"

os.makedirs(MODEL_DIR, exist_ok=True)

class PromptSimplifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def simplify_prompt(self, prompt: str) -> str:
        input_ids = self.tokenizer.encode("simplify: " + prompt, return_tensors="pt", truncation=True)
        output = self.model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def extract_features(self, text: str) -> dict:
        return {
            "token_count": len(self.tokenizer.tokenize(text)),
            "readability_score": flesch_reading_ease(text)
        }

    def save_all(self):
        # Save .pt
        torch.save(self.model.state_dict(), os.path.join(MODEL_DIR, "model.pt"))

        # Save .pkl
        with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
            pickle.dump(self.tokenizer, f)

        # Save .h5 (only possible if using Keras-compatible model)
        try:
            self.model.save_pretrained(MODEL_DIR, safe_serialization=True)
            print("Saved using HuggingFace format (Safe Serialization).")
        except Exception as e:
            print("H5 saving failed:", e)


# Example usage
if __name__ == "__main__":
    simplifier = PromptSimplifier()

    original_prompt = "Explain the technological implications of artificial intelligence in layman's terms."
    simplified_prompt = simplifier.simplify_prompt(original_prompt)

    original_features = simplifier.extract_features(original_prompt)
    simplified_features = simplifier.extract_features(simplified_prompt)

    print("Original Prompt:\n", original_prompt)
    print("Simplified Prompt:\n", simplified_prompt)
    print("\nOriginal Features:", original_features)
    print("Simplified Features:", simplified_features)

    # Save model and tokenizer
    simplifier.save_all()
