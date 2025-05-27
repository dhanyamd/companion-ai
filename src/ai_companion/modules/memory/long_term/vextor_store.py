import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import List, Optional
import logging
from pathlib import Path

from settings import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Define model paths
MODEL_CACHE_DIR = Path("models")
EMBEDDING_MODEL = "all-MiniLM-L3-v2"  # Smaller, faster model
EMBEDDING_MODEL_PATH = MODEL_CACHE_DIR / EMBEDDING_MODEL

@dataclass
class Memory:
    """Represents a memory entry in the vector store."""

    text: str
    metadata: dict
    score: Optional[float] = None

    @property
    def id(self) -> Optional[str]:
        return self.metadata.get("id")

    @property
    def timestamp(self) -> Optional[datetime]:
        ts = self.metadata.get("timestamp")
        return datetime.fromisoformat(ts) if ts else None


class VectorStore:
    """A class to handle vector storage operations using Qdrant."""

    REQUIRED_ENV_VARS = ["QDRANT_URL", "QDRANT_API_KEY"]
    EMBEDDING_MODEL = EMBEDDING_MODEL
    COLLECTION_NAME = "long_term_memory"
    SIMILARITY_THRESHOLD = 0.9  # Threshold for considering memories as similar

    _instance: Optional["VectorStore"] = None
    _initialized: bool = False
    _initialization_error: Optional[Exception] = None

    def __new__(cls) -> "VectorStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            if self._initialization_error:
                raise self._initialization_error
            return

        try:
            self._validate_env_vars()
            
            # Create model cache directory if it doesn't exist
            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
            # Try to load from local cache first
            if EMBEDDING_MODEL_PATH.exists():
                logger.info(f"Loading model from local cache: {EMBEDDING_MODEL_PATH}")
                try:
                    self.model = SentenceTransformer(str(EMBEDDING_MODEL_PATH))
                    logger.info("Successfully loaded model from cache")
                except Exception as e:
                    logger.warning(f"Failed to load model from cache: {str(e)}")
                    self._download_and_save_model()
            else:
                self._download_and_save_model()
            
            logger.info("Initializing Qdrant client...")
            self.client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
            self._initialized = True
            logger.info("Vector store initialized successfully")
        except Exception as e:
            self._initialization_error = e
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    def _download_and_save_model(self) -> None:
        """Download and save the model locally."""
        try:
            logger.info(f"Downloading model {self.EMBEDDING_MODEL}...")
            self.model = SentenceTransformer(self.EMBEDDING_MODEL)
            logger.info(f"Saving model to {EMBEDDING_MODEL_PATH}")
            self.model.save(str(EMBEDDING_MODEL_PATH))
            logger.info("Model downloaded and saved successfully")
        except Exception as e:
            logger.error(f"Failed to download and save model: {str(e)}")
            raise

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def _collection_exists(self) -> bool:
        """Check if the memory collection exists."""
        collections = self.client.get_collections().collections
        return any(col.name == self.COLLECTION_NAME for col in collections)

    def _create_collection(self) -> None:
        """Create a new collection for storing memories."""
        sample_embedding = self.model.encode("sample text")
        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=len(sample_embedding),
                distance=Distance.COSINE,
            ),
        )

    def find_similar_memory(self, text: str) -> Optional[Memory]:
        """Find if a similar memory already exists.

        Args:
            text: The text to search for

        Returns:
            Optional Memory if a similar one is found
        """
        results = self.search_memories(text, k=1)
        if results and results[0].score >= self.SIMILARITY_THRESHOLD:
            return results[0]
        return None

    def store_memory(self, text: str, metadata: dict) -> None:
        """Store a new memory in the vector store or update if similar exists.

        Args:
            text: The text content of the memory
            metadata: Additional information about the memory (timestamp, type, etc.)
        """
        if not self._collection_exists():
            self._create_collection()

        # Check if similar memory exists
        similar_memory = self.find_similar_memory(text)
        if similar_memory and similar_memory.id:
            metadata["id"] = similar_memory.id  # Keep same ID for update

        embedding = self.model.encode(text)
        point = PointStruct(
            id=metadata.get("id", hash(text)),
            vector=embedding.tolist(),
            payload={
                "text": text,
                **metadata,
            },
        )

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point],
        )

    def search_memories(self, query: str, k: int = 5) -> List[Memory]:
        """Search for similar memories in the vector store.

        Args:
            query: Text to search for
            k: Number of results to return

        Returns:
            List of Memory objects
        """
        if not self._collection_exists():
            return []

        query_embedding = self.model.encode(query)
        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=k,
        )

        return [
            Memory(
                text=hit.payload["text"],
                metadata={k: v for k, v in hit.payload.items() if k != "text"},
                score=hit.score,
            )
            for hit in results
        ]


@lru_cache
def get_vector_store() -> VectorStore:
    """Get or create the VectorStore singleton instance."""
    try:
        return VectorStore()
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise