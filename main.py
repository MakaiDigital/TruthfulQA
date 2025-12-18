import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


import logging

from src.config import Config
from src.evaluation.evaluate import RAGEvaluator
from src.pipeline.ingestion import DataIngestion
from src.pipeline.processing import DataProcessor
from src.pipeline.vectorization2 import EnhancedVectorStore
from src.rag.hybrid_rag import AdvancedRAGSystem
from src.rag.retriever import HybridRetriever
from src.utils.cache import CacheManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def run_advanced_pipeline():

    config = Config()

    # Initialize cache
    logging.info("Initializing cache system...")
    cache_manager = CacheManager(config)

    # Ingest data
    logging.info(" Loading dataset...")
    ingestion = DataIngestion(config.DATA_DIR)
    df = ingestion.load_dataset()

    # Validate schema
    if not ingestion.validate_schema(df):
        logging.error("Dataset validation failed!")
        return None, None

    logging.info(f"Loaded {len(df)} questions")

    #  Process and filter
    logging.info("Step 2: Processing and filtering data...")
    processor = DataProcessor()
    documents = processor.filter_and_clean(df)

    if not documents:
        logging.error(" No documents extracted! Check data format.")
        sys.exit(1)

    logging.info(f"Extracted {len(documents)} original documents")

    logging.info(" Generating embeddings and indexing...")
    vector_store = EnhancedVectorStore(
        model_name=config.EMBEDDING_MODEL,
        db_path=config.VECTOR_DB_DIR,
        collection_name=config.COLLECTION_NAME,
    )

    # create/get collection first
    vector_store.get_or_create_collection()

    # Check if already indexed
    existing_docs = vector_store.load_documents()
    if existing_docs and len(existing_docs) == len(documents):
        logging.info("Using existing vector index")
        documents = existing_docs
    else:
        logging.info("Creating new vector index...")
        vector_store.index_documents(documents)

    #  Initialize  retriever
    logging.info("Initializing  retriever...")
    hybrid_retriever = HybridRetriever(vector_store, documents, config)

    # Initialize Advanced RAG
    logging.info(" Initializing advanced RAG system...")
    rag_system = AdvancedRAGSystem(hybrid_retriever, config, cache_manager)

    # Evaluate
    logging.info(" Running evaluation...")
    evaluator = RAGEvaluator(rag_system, config.EMBEDDING_MODEL, df, config)
    results = evaluator.evaluate()

    # Save results
    config.RESULTS_DIR.mkdir(exist_ok=True)
    output_path = config.RESULTS_DIR / "results.csv"
    evaluator.save_results(results, output_path)

    logging.info(f"\n Pipeline complete! Results saved to {output_path}")

    return rag_system, results


if __name__ == "__main__":
    try:
        rag_system, results = run_advanced_pipeline()
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
