
import os
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import T5ForConditionalGeneration, T5Tokenizer


class PromptOptimizer:
    def __init__(self,
                 embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
                 generator_model_name='ramsrigouthamg/t5_paraphraser',
                 save_dir="model/prompt_optimizer",
                 device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir

        self.embedding_model_path = os.path.join(save_dir, "sentence_transformer")
        self.generator_model_path = os.path.join(save_dir, "t5_paraphraser")

        # Load or initialize models
        self.embedding_model = self.load_or_init_embedding_model(embedding_model_name)
        self.tokenizer, self.generator_model = self.load_or_init_generator(generator_model_name)

    def load_or_init_embedding_model(self, model_name):
        if os.path.exists(self.embedding_model_path):
            print("✅ Loading saved SentenceTransformer...")
            return SentenceTransformer(self.embedding_model_path)
        else:
            print("⬇️ Downloading and saving SentenceTransformer...")
            model = SentenceTransformer(model_name)
            model.save(self.embedding_model_path)
            return model

    def load_or_init_generator(self, model_name):
        if os.path.exists(self.generator_model_path):
            print("✅ Loading saved T5 paraphraser model...")
            tokenizer = T5Tokenizer.from_pretrained(self.generator_model_path)
            model = T5ForConditionalGeneration.from_pretrained(self.generator_model_path).to(self.device)
        else:
            print("⬇️ Downloading and saving T5 paraphraser model...")
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
            tokenizer.save_pretrained(self.generator_model_path)
            model.save_pretrained(self.generator_model_path)
        return tokenizer, model

    def generate_variants(self, prompt, num_variants=5, max_length=64):
        input_text = f"paraphrase: {prompt} </s>"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        outputs = self.generator_model.generate(
            input_ids=input_ids,
            do_sample=True,
            max_length=max_length,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=num_variants
        )

        variants = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for output in outputs]
        return list(set(variants))

    def rank_variants(self, original_prompt, variants, top_k=3):
        original_embedding = self.embedding_model.encode(original_prompt, convert_to_tensor=True)
        variant_embeddings = self.embedding_model.encode(variants, convert_to_tensor=True)

        similarities = util.cos_sim(original_embedding, variant_embeddings)[0]
        ranked = sorted(zip(variants, similarities), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def suggest_prompts(self, prompt, num_variants=5, top_k=3):
        variants = self.generate_variants(prompt, num_variants=num_variants)
        ranked_variants = self.rank_variants(prompt, variants, top_k=top_k)
        return ranked_variants


# Example usage
if __name__ == "__main__":
    prompt = "Explain how quantum computing will change artificial intelligence."

    optimizer = PromptOptimizer()
    suggestions = optimizer.suggest_prompts(prompt, num_variants=10, top_k=5)

    print(f"\nOriginal Prompt:\n{prompt}\n")
    print("Suggested Low-Energy Alternatives:\n")
    for idx, (variant, score) in enumerate(suggestions, 1):
        print(f"{idx}. {variant}  [Similarity: {score.item():.4f}]")
