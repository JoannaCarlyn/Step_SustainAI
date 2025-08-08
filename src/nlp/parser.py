# src/nlp/parser.py

import re
import nltk

nltk.download('punkt')

def clean_prompt(prompt: str) -> str:
    """
    Cleans the input prompt by removing extra whitespace, special characters, etc.
    """
    prompt = prompt.strip()
    prompt = re.sub(r'\s+', ' ', prompt)
    return prompt

def tokenize_prompt(prompt: str) -> list:
    """
    Tokenizes the prompt using NLTK word tokenizer.
    """
    return nltk.word_tokenize(prompt)
