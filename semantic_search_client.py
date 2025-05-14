import os
import logging
from typing import Optional, List
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, Index
from dotenv import load_dotenv


class Config:
    PINECONE_INDEX_NAME = "brightdata-ai-dataset"  # Pinecone index name
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # SentenceTransformer model name
    DEFAULT_TOP_K = 3  # Number of results to return


def configure_logging():
    """Configures global logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


logger = configure_logging()
load_dotenv()


class SearchService:
    def __init__(self):
        self.index: Optional[Index] = None
        self.encoder: Optional[SentenceTransformer] = None

    def initialize_services(self) -> bool:
        """Initializes Pinecone and embedding model services"""
        try:
            self._init_encoder()
            self._init_pinecone()
            return True
        except Exception as e:
            logger.critical(f"Service initialization failed: {str(e)}")
            return False

    def _init_pinecone(self):
        """Initializes Pinecone connection and validates index"""
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key or api_key.startswith("YOUR_"):
            raise ValueError("Invalid Pinecone API key configuration")

        pc = Pinecone(api_key=api_key)
        self.index = pc.Index(Config.PINECONE_INDEX_NAME)
        self._validate_index()

    def _validate_index(self):
        """Validates index configuration matches expectations"""
        if not self.encoder:
            raise ValueError("Encoder not initialized before index validation")

        stats = self.index.describe_index_stats()
        logger.info(f"Index stats: {stats}")

        encoder_dim = self.encoder.get_sentence_embedding_dimension()
        if stats.dimension != encoder_dim:
            raise ValueError(
                f"Index dimension {stats.dimension} does not match "
                f"encoder dimension {encoder_dim}"
            )

    def _init_encoder(self):
        """Initializes the sentence embedding model"""
        self.encoder = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        logger.info(f"Encoder model initialized: {self.encoder}")

    def semantic_search(
        self, query: str, top_k: int = Config.DEFAULT_TOP_K
    ) -> List[dict]:
        """Performs semantic search with error handling"""
        if not query.strip():
            logger.warning("Empty query received")
            return []

        try:
            embedding = self.encoder.encode(query, show_progress_bar=False).tolist()
            results = self.index.query(
                vector=embedding, top_k=top_k, include_metadata=True
            )
            return results.get("matches", [])
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []


class ResultPresenter:
    @staticmethod
    def display(results: List[dict]):
        """Displays search results in a consistent format"""
        if not results:
            print("\nNo results found")
            return

        print("\n--- Search Results ---")
        for i, match in enumerate(results, 1):
            metadata = match.get("metadata", {})
            print(f"\n#{i} (Score: {match.score:.4f})")
            print(f"ID: {match.id}")
            print(f"Date: {metadata.get('date', 'N/A')}")
            print(f"Rating: {metadata.get('rating', 'N/A')}")

            if title := metadata.get("title"):
                print(f"Title: {title}")

            if description := metadata.get("description"):
                print(f"Description: {description}")


def example_queries(service: SearchService):
    """Example query demonstration"""
    queries = [
        "good price for students",
        "excellent battery life",
        "issues with shipping or delivery",
        "lightweight and good for travel",
        "terrible product, waste of money",
    ]

    for query in queries:
        logger.info(f"\nProcessing query: {query}")
        results = service.semantic_search(query, top_k=3)
        ResultPresenter.display(results)
        print("\n" + "=" * 50)


def main():
    logger.info("🚀 Starting Semantic Search Service")

    service = SearchService()
    if not service.initialize_services():
        logger.critical("❌ Service initialization failed. Exiting.")
        return

    example_queries(service)
    logger.info("✅ Service completed successfully")


if __name__ == "__main__":
    main()