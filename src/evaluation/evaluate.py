import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RAGEvaluator:
    def __init__(self, rag_system, embedding_model, dataset: pd.DataFrame, config):
        self.rag_system = rag_system
        self.model = SentenceTransformer(embedding_model)
        self.dataset = dataset
        self.config = config

    def select_sample_questions(self, n: int = 20) -> List[Dict]:
        "Select random questions for evaluation"
        sample = self.dataset.sample(n=min(n, len(self.dataset)))
        return [
            {"question": row["Question"], "best_answer": row["Best Answer"]}
            for _, row in sample.iterrows()
        ]

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        emb1 = self.model.encode([text1])
        emb2 = self.model.encode([text2])
        return cosine_similarity(emb1, emb2)[0][0]

    def evaluate(self) -> pd.DataFrame:
        """Run evaluation on sample questions"""
        sample_questions = self.select_sample_questions(self.config.EVAL_SAMPLE_SIZE)
        results = []

        for item in sample_questions:
            question = item["question"]
            best_answer = item["best_answer"]

            # Generate answer using RAG
            rag_response = self.rag_system.query(question)
            generated_answer = rag_response["answer"]

            # Calculate similarity
            similarity = self.calculate_similarity(generated_answer, best_answer)

            results.append(
                {
                    "Question": question,
                    "Generated Answer": generated_answer,
                    "Best Answer": best_answer,
                    "Similarity Score": similarity,
                }
            )

        return pd.DataFrame(results)

    def save_results(self, results: pd.DataFrame, output_path: Path):
        """Save results to CSV"""
        results.to_csv(output_path, index=False)

        # Print summary statistics
        print(f"\n=== Evaluation Summary ===")
        print(f"Average Similarity: {results['Similarity Score'].mean():.3f}")
        print(f"Median Similarity: {results['Similarity Score'].median():.3f}")
        print(f"Min Similarity: {results['Similarity Score'].min():.3f}")
        print(f"Max Similarity: {results['Similarity Score'].max():.3f}")
