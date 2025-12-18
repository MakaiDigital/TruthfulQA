import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class EnhancedVectorStore:
    def __init__(self, model_name: str, db_path: Path, collection_name: str):
        self.model = SentenceTransformer(model_name)
        self.db_path = db_path
        self.client = chromadb.PersistentClient(
            path=str(db_path), settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = collection_name
        self.collection = None
        self.documents = []  # Store for BM25

    def create_collection(self):
        """Create or get collection"""
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            logging.info(f"Created new collection: {self.collection_name}")
        except:
            self.collection = self.client.get_collection(self.collection_name)
            logging.info(f"Using existing collection: {self.collection_name}")

    def get_or_create_collection(self):
        """Get existing collection or create new one"""
        if self.collection is not None:
            return self.collection

        try:
            # Try to get existing collection first
            self.collection = self.client.get_collection(self.collection_name)
            logging.info(f"Loaded existing collection: {self.collection_name}")
        except:
            # Create new if doesn't exist
            self.collection = self.client.create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            logging.info(f"Created new collection: {self.collection_name}")

        return self.collection

    def index_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """Generate embeddings and store in ChromaDB"""
        self.get_or_create_collection()

        self.documents = documents  # Store for hybrid retrieval

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
        logging.info(f"Generating embeddings for {len(texts)} documents...")
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_metadata = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            # Generate embeddings
            embeddings = self.model.encode(
                batch_texts, show_progress_bar=True, batch_size=batch_size
            )

            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=batch_texts,
                metadatas=batch_metadata,
                ids=batch_ids,
            )

        # Save documents for BM25
        self._save_documents(documents)

        logging.info(f"✓ Indexed {len(documents)} documents")

    def _save_documents(self, documents: List[Dict[str, Any]]):
        """Save documents for BM25 retrieval"""
        doc_path = self.db_path / "documents.pkl"
        with open(doc_path, "wb") as f:
            pickle.dump(documents, f)
        logging.info(f"Saved documents to {doc_path}")

    def load_documents(self) -> List[Dict[str, Any]]:
        """Load documents for BM25 retrieval"""
        doc_path = self.db_path / "documents.pkl"
        if doc_path.exists():
            with open(doc_path, "rb") as f:
                documents = pickle.load(f)
            logging.info(f"Loaded {len(documents)} documents from cache")
            # IMPORTANT: Also load the collection when loading documents
            self.get_or_create_collection()
            return documents
        return []

    def search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Retrieve top-k relevant documents"""
        # Ensure collection is loaded
        if self.collection is None:
            self.get_or_create_collection()

        query_embedding = self.model.encode([query])

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(), n_results=top_k
        )

        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
        }
