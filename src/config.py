import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Paths
    DATA_DIR: Path = Path("data")
    VECTOR_DB_DIR: Path = Path("vector_db")
    RESULTS_DIR: Path = Path("results")
    MODELS_DIR: Path = Path("models")
    CACHE_DIR: Path = Path("cache")

    # Model settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CHUNK_SIZE: int = 512

    # Retrieval settings
    TOP_K: int = 10  # Increased for hybrid search
    TOP_K_RERANK: int = 3  # Final results after reranking
    DENSE_WEIGHT: float = 0.7  # Weight for dense retrieval
    SPARSE_WEIGHT: float = 0.3  # Weight for sparse retrieval

    # Query expansion
    ENABLE_QUERY_EXPANSION: bool = False
    NUM_EXPANSIONS: int = 2

    # Reranking
    ENABLE_RERANKING: bool = True
    RERANK_TOP_K: int = 20  # Number of docs to rerank

    # Metadata filtering
    FILTER_BY_SOURCE: Optional[List[str]] = None
    FILTER_BY_ANSWER_TYPE: Optional[List[str]] = None

    # Caching
    ENABLE_CACHE: bool = False
    CACHE_TYPE: str = os.getenv("CACHE_TYPE", "disk")  # disk, redis, memory
    CACHE_TTL: int = 3600  # 1 hour
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))

    # LLM settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 200

    # Vector DB
    COLLECTION_NAME: str = "truthful_qa"

    # Evaluation
    EVAL_SAMPLE_SIZE: int = 20
    SIMILARITY_THRESHOLD: float = 0.7

    def __post_init__(self):
        # Create directories
        for dir_path in [
            self.DATA_DIR,
            self.VECTOR_DB_DIR,
            self.RESULTS_DIR,
            self.MODELS_DIR,
            self.CACHE_DIR,
        ]:
            dir_path.mkdir(exist_ok=True, parents=True)
