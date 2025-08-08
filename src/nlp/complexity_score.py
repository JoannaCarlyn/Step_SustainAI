# src/nlp/feature_extractor.py

from transformers import AutoTokenizer
import nltk
import textstat

nltk.download('punkt')  # Required for textstat

# Load transformer tokenizer (BERT used here)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def compute_token_count(prompt: str) -> int:
    tokens = tokenizer(prompt)["input_ids"]
    return len(tokens)

def compute_readability_score(prompt: str) -> float:
    return textstat.flesch_reading_ease(prompt)

def extract_features(prompt: str) -> dict:
    return {
        "token_count": compute_token_count(prompt),
        "readability_score": compute_readability_score(prompt)
    }

# Example usage
if __name__ == "__main__":
    sample_prompt = "Write a comprehensive explanation of the impacts of AI on society."
    features = extract_features(sample_prompt)
    print("Extracted Features:", features)
# src/nlp/complexity_score.py

from transformers import AutoTokenizer
import textstat

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def compute_token_count(prompt: str) -> int:
    """
    Uses HuggingFace tokenizer to compute number of tokens.
    """
    tokens = tokenizer(prompt)["input_ids"]
    return len(tokens)

def compute_readability(prompt: str) -> float:
    """
    Computes readability score (Flesch Reading Ease).
    """
    return textstat.flesch_reading_ease(prompt)

def compute_complexity_features(prompt: str) -> dict:
    """
    Combines token count and readability into one dict.
    """
    return {
        "token_count": compute_token_count(prompt),
        "readability_score": compute_readability(prompt)
    }
