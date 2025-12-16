import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    def __init__(self, vector_store, documents: List[Dict[str, Any]], config):
        self.vector_store = vector_store
        self.config = config
        self.documents = documents

        # Initialize BM25
        self.corpus = [doc["text"] for doc in documents]
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Document lookup
        self.doc_lookup = {i: doc for i, doc in enumerate(documents)}

        logging.info("Hybrid retriever initialized with BM25 and dense search")

    def dense_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Dense vector search using embeddings"""
        results = self.vector_store.search(query, top_k=top_k)

        dense_results = []
        for doc_text, metadata, distance in zip(
            results["documents"], results["metadatas"], results["distances"]
        ):
            dense_results.append(
                {
                    "text": doc_text,
                    "metadata": metadata,
                    "score": 1 - distance,  # Convert distance to similarity
                    "method": "dense",
                }
            )

        return dense_results

    def sparse_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Sparse search using BM25"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        sparse_results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                doc = self.doc_lookup[idx]
                sparse_results.append(
                    {
                        "text": doc["text"],
                        "metadata": {
                            "question": doc["question"],
                            "answer_type": doc["answer_type"],
                            "source": doc["source"],
                        },
                        "score": float(scores[idx]),
                        "method": "sparse",
                    }
                )

        return sparse_results

    def hybrid_search(
        self,
        query: str,
        top_k: int,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Combine dense and sparse search with weighted scores"""
        dense_weight = dense_weight or self.config.DENSE_WEIGHT
        sparse_weight = sparse_weight or self.config.SPARSE_WEIGHT

        # Perform both searches
        dense_results = self.dense_search(query, top_k * 2)
        sparse_results = self.sparse_search(query, top_k * 2)

        # Normalize scores
        dense_results = self._normalize_scores(dense_results)
        sparse_results = self._normalize_scores(sparse_results)

        # Combine results using Reciprocal Rank Fusion
        combined_scores = defaultdict(lambda: {"score": 0, "doc": None})

        # Add dense results
        for rank, result in enumerate(dense_results):
            doc_key = result["text"]
            combined_scores[doc_key]["score"] += dense_weight * result["score"]
            combined_scores[doc_key]["doc"] = result

        # Add sparse results
        for rank, result in enumerate(sparse_results):
            doc_key = result["text"]
            if combined_scores[doc_key]["doc"] is None:
                combined_scores[doc_key]["doc"] = result
            combined_scores[doc_key]["score"] += sparse_weight * result["score"]
            combined_scores[doc_key]["doc"]["method"] = "hybrid"

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(), key=lambda x: x["score"], reverse=True
        )

        # Return top k
        return [item["doc"] for item in sorted_results[:top_k]]

    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Min-max normalization of scores"""
        if not results:
            return results

        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score - min_score == 0:
            return results

        for result in results:
            result["score"] = (result["score"] - min_score) / (max_score - min_score)

        return results

    def search_with_filter(
        self,
        query: str,
        top_k: int,
        source_filter: Optional[List[str]] = None,
        answer_type_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search with metadata filtering"""
        # Get more results than needed for filtering
        results = self.hybrid_search(query, top_k * 3)

        # Apply filters
        filtered_results = []
        for result in results:
            metadata = result["metadata"]

            # Source filter
            if source_filter and metadata.get("source") not in source_filter:
                continue

            # Answer type filter
            if answer_type_filter:
                answer_type = metadata.get("answer_type", "")
                if not any(at in answer_type for at in answer_type_filter):
                    continue

            filtered_results.append(result)

            if len(filtered_results) >= top_k:
                break

        return filtered_results
