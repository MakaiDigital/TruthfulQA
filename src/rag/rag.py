import logging
from typing import Any, Dict, List

import openai

from src.config import Config
from src.pipeline.vectorization import VectorStore


class RAGSystem:
    def __init__(self, vector_store: VectorStore, config: Config):
        self.vector_store = vector_store
        self.config = config

        # Initialize LLM
        if config.LLM_PROVIDER == "openai":
            openai.api_key = config.OPENAI_API_KEY

    def retrieve_context(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant context from vector DB"""
        k = top_k or self.config.TOP_K
        results = self.vector_store.search(query, top_k=k)

        contexts = []
        for doc, metadata, distance in zip(
            results["documents"], results["metadatas"], results["distances"]
        ):
            contexts.append(
                {
                    "text": doc,
                    "question": metadata["question"],
                    "source": metadata["source"],
                    "similarity": 1 - distance,  # Convert distance to similarity
                }
            )

        return contexts

    def generate_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Generate answer using LLM with retrieved context"""
        # Build context string
        context_str = "\n\n".join(
            [
                f"Context {i+1}: {ctx['text']}\n(Related to: {ctx['question']})"
                for i, ctx in enumerate(contexts)
            ]
        )

        # Create prompt
        prompt = f"""You are a truthful assistant. Answer the question based only on the provided context.

Context:
{context_str}

Question: {query}

Answer truthfully and concisely based on the context above. If the context doesn't contain the answer, say "I don't have enough information to answer this question."

Answer:"""

        try:
            if self.config.LLM_PROVIDER == "openai":
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a truthful assistant that answers questions based on provided context.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=200,
                )
                return response.choices[0].message.content
            else:
                # Mock generator fallback
                return self._mock_generate(contexts)
        except Exception as e:
            logging.error(f"LLM generation failed: {e}")
            return self._mock_generate(contexts)

    def _mock_generate(self, contexts: List[Dict[str, Any]]) -> str:
        """Fallback mock generator"""
        if not contexts:
            return "I don't have enough information to answer this question."

        return f"{contexts[0]['text']}"

    def query(self, question: str) -> Dict[str, Any]:
        """Full RAG pipeline"""
        contexts = self.retrieve_context(question)
        answer = self.generate_answer(question, contexts)

        return {"question": question, "answer": answer, "contexts": contexts}
