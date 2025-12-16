import logging
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, model_name: str, db_path: Path, collection_name: str):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(
            path=str(db_path), settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = collection_name
        self.collection = None

    def create_collection(self):
        """Create or get collection"""
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
        except:
            self.collection = self.client.get_collection(self.collection_name)

    def index_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """Generate embeddings and store in ChromaDB"""
        if not self.collection:
            self.create_collection()

        texts = [doc["text"] for doc in documents]
        metadatas = [
            {
                "question": doc["question"],
                "answer_type": doc["answer_type"],
                "source": doc["source"],
            }
            for doc in documents
        ]
        ids = [f"doc_{i}" for i in range(len(documents))]

        # Batch processing for efficiency
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_metadata = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            # Generate embeddings
            embeddings = self.model.encode(batch_texts, show_progress_bar=True)

            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=batch_texts,
                metadatas=batch_metadata,
                ids=batch_ids,
            )

        logging.info(f"Indexed {len(documents)} documents")

    def search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Retrieve top-k relevant documents"""
        query_embedding = self.model.encode([query])

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(), n_results=top_k
        )

        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
        }
