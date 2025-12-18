import logging
from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str, config):
        self.config = config
        self.model = CrossEncoder(model_name)
        logging.info(f"Reranker initialized: {model_name}")

    def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder"""
        if not self.config.ENABLE_RERANKING or not documents:
            return documents[:top_k] if top_k else documents

        top_k = top_k or self.config.TOP_K_RERANK

        # Prepare query-document pairs
        pairs = [[query, doc["text"]] for doc in documents]

        # Get reranking scores
        scores = self.model.predict(pairs)

        # Add reranking scores to documents
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # Sort by reranking score
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        logging.info(f"Reranked {len(documents)} documents, returning top {top_k}")

        return reranked[:top_k]

    def rerank_with_threshold(
        self, query: str, documents: List[Dict[str, Any]], threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Rerank and filter by minimum score threshold"""
        reranked = self.rerank(query, documents, top_k=None)
        return [doc for doc in reranked if doc.get("rerank_score", 0) >= threshold]
