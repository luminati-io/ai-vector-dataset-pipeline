import json
import time
import os
import uuid
import logging
from typing import List, Dict, Optional, Union
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, Index
from dotenv import load_dotenv


# Configuration class for centralizing settings
class Config:
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Name of the embedding model
    EMBEDDING_DIMENSION = 384  # Dimension of the embedding model
    PINECONE_INDEX_NAME = "brightdata-ai-dataset"  # Name of the Pinecone index
    INPUT_JSON_FILE = "extracted_walmart_reviews.json"  # Path to the input JSON file
    UPSERT_BATCH_SIZE = 100  # Batch size for upsert operations
    EMBEDDING_BATCH_SIZE = 32  # Batch size for embedding generation
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    LOG_FILE = "data_pipeline.log"


# Initialize logging once at module level
logging.basicConfig(
    level=logging.INFO,
    format=Config.LOG_FORMAT,
    handlers=[logging.FileHandler(Config.LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

load_dotenv()


class DataPipeline:
    def __init__(self):
        self.pinecone_index: Optional[Index] = None
        self.embedding_model: Optional[SentenceTransformer] = None

    def load_reviews(self) -> List[Dict]:
        """Load and validate review data from JSON file"""
        try:
            with open(Config.INPUT_JSON_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("Invalid JSON structure - expected list of reviews")

            logger.info(f"Loaded {len(data)} reviews from {Config.INPUT_JSON_FILE}")
            return data

        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to load reviews: {str(e)}")
            raise SystemExit(1)

    def initialize_models(self):
        """Load and validate embedding model"""
        try:
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
            if (
                self.embedding_model.get_sentence_embedding_dimension()
                != Config.EMBEDDING_DIMENSION
            ):
                raise ValueError("Model dimension mismatch")

            logger.info(f"Initialized embedding model: {Config.EMBEDDING_MODEL_NAME}")

        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise SystemExit(1)

    def generate_embeddings(self, reviews: List[Dict]) -> List[Dict]:
        """Batch process embeddings for all reviews with parallel processing"""
        texts = []
        valid_indices = []

        for idx, review in enumerate(reviews):
            text = self._build_review_text(review)
            if text:
                texts.append(text)
                valid_indices.append(idx)

        logger.info(f"Generating embeddings for {len(texts)} valid reviews")
        start = time.time()

        # Enable parallel processing with batching
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=Config.EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True,
        ).tolist()

        elapsed = time.time() - start

        # Map embeddings back to original reviews
        for emb_idx, review_idx in enumerate(valid_indices):
            reviews[review_idx]["embedding"] = embeddings[emb_idx]

        logger.info(
            f"Generated {len(embeddings)} embeddings in {elapsed:.1f} seconds "
            f"({len(texts)/max(elapsed, 0.001):.1f} reviews/sec)"
        )
        return reviews

    def _build_review_text(self, review: Dict) -> str:
        """Construct combined text from review components"""
        title = str(review.get("title") or "").strip()
        description = str(review.get("description") or "").strip()

        parts = []
        if title:
            parts.append(f"Review Title: {title}")
        if description:
            parts.append(f"Review Description: {description}")

        return ". ".join(parts) if parts else ""

    def connect_pinecone(self):
        """Initialize and validate Pinecone connection"""
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key or api_key.startswith("YOUR_"):
            logger.error("Missing Pinecone API key in environment")
            raise SystemExit(1)

        try:
            pc = Pinecone(api_key=api_key)
            self.pinecone_index = pc.Index(Config.PINECONE_INDEX_NAME)

            # Validate index dimensions
            stats = self.pinecone_index.describe_index_stats()
            if stats.dimension != Config.EMBEDDING_DIMENSION:
                raise ValueError("Index dimension mismatch")

            logger.info(f"Connected to Pinecone index: {Config.PINECONE_INDEX_NAME}")

        except Exception as e:
            logger.error(f"Pinecone connection failed: {str(e)}")
            raise SystemExit(1)

    def prepare_vectors(self, reviews: List[Dict]) -> List[Dict]:
        """Create Pinecone vectors from enriched reviews"""
        vectors = []
        for review in reviews:
            if not review.get("embedding"):
                continue

            vectors.append(
                {
                    "id": str(uuid.uuid4()),
                    "values": review["embedding"],
                    "metadata": self._sanitize_metadata(review),
                }
            )

        logger.info(f"Prepared {len(vectors)} vectors for upsert")
        return vectors

    def _sanitize_metadata(self, review: Dict) -> Dict:
        """Clean and format metadata for Pinecone storage, omitting null/empty values"""
        sanitized = {}
        for key, value in review.items():
            if key == "embedding":
                continue

            formatted_value = self._format_metadata_value(value)
            if formatted_value is not None:
                sanitized[key] = formatted_value

        return sanitized

    def _format_metadata_value(
        self, value
    ) -> Optional[Union[str, int, float, bool, list]]:
        """Ensure metadata values are Pinecone-compatible, returning None for invalid values"""
        if value is None:
            return None
        if isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, str):
            return value.strip() if value.strip() else None
        if isinstance(value, list):
            return [str(item) for item in value] if value else None
        return str(value)

    def upsert_vectors(self, vectors: List[Dict]):
        """Batch upsert vectors to Pinecone"""
        total_upserted = 0
        for batch in self._batch_generator(vectors, Config.UPSERT_BATCH_SIZE):
            try:
                response = self.pinecone_index.upsert(vectors=batch)
                total_upserted += len(batch)
                logger.debug(f"Upserted batch of {len(batch)} vectors")
            except Exception as e:
                logger.error(f"Batch upsert failed: {str(e)}")

        logger.info(f"Successfully upserted {total_upserted} vectors")

    def _batch_generator(self, data: List, batch_size: int):
        """Generate data in fixed-size chunks"""
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]


def main():
    pipeline = DataPipeline()

    # Execution flow
    reviews = pipeline.load_reviews()
    pipeline.initialize_models()
    embedded_reviews = pipeline.generate_embeddings(reviews)
    pipeline.connect_pinecone()

    vectors = pipeline.prepare_vectors(embedded_reviews)
    if vectors:
        pipeline.upsert_vectors(vectors)

    logger.info("Pipeline execution completed successfully")


if __name__ == "__main__":
    main()