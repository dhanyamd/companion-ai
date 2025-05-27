import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import List, Optional
import logging
import re
from pathlib import Path
import time
import requests
from huggingface_hub import HfFolder, hf_hub_download, HfApi
from huggingface_hub.utils import HfHubHTTPError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from settings import settings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer 
from ai_companion.core.utils import clean_url

logger = logging.getLogger(__name__)

# Define model paths
MODEL_CACHE_DIR = Path("models")
EMBEDDING_MODEL = "all-MiniLM-L3-v2"  # Changed to a smaller model
EMBEDDING_MODEL_PATH = MODEL_CACHE_DIR / EMBEDDING_MODEL

def check_huggingface_connection() -> bool:
    """Check if we can connect to Hugging Face."""
    try:
        response = requests.get("https://huggingface.co", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Failed to connect to Hugging Face: {str(e)}")
        return False

def verify_model_files(model_path: Path) -> bool:
    """Verify that all required model files exist."""
    required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
    return all((model_path / file).exists() for file in required_files)

# Configure Hugging Face token if available
if os.getenv("HUGGINGFACE_TOKEN"):
    HfFolder.save_token(os.getenv("HUGGINGFACE_TOKEN"))

# Configure retry strategy for requests
retry_strategy = Retry(
    total=3,  # number of retries
    backoff_factor=2,  # wait 2, 4, 8 seconds between retries
    status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

def clean_api_key(key: str) -> str:
    """Clean API key by removing all non-alphanumeric characters except for dots."""
    # Remove all whitespace and newlines
    key = ''.join(key.split())
    # Remove any non-alphanumeric characters except dots
    key = re.sub(r'[^a-zA-Z0-9.]', '', key)
    return key

@dataclass 
class Memory: 
    """Represents a memory entry in the vector store. """
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
    """A class to handle vector storage operations using Qdrant. """
    
    REQUIRED_ENV_VARS = ["QDRANT_URL", "QDRANT_API_KEY"]
    EMBEDDING_MODEL = EMBEDDING_MODEL
    COLLECTION_NAME = "long_term_memory"
    SIMILARITY_THRESHOLD = 0.9 
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    FALLBACK_MODELS = [
        "all-MiniLM-L3-v2",  # Primary model (smaller, faster)
        "paraphrase-MiniLM-L3-v2",  # First fallback
        "distiluse-base-multilingual-cased-v1"  # Second fallback
    ]

    _instance: Optional["VectorStore"] = None 
    _initialized: bool = False

    def __new__(cls) -> "VectorStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance 
    
    def __init__(self) -> None:
        """Initialize the vector store with Qdrant client."""
        if self._initialized:
            return

        try:
            # Check Hugging Face connection first
            if not check_huggingface_connection():
                raise EnvironmentError("Cannot connect to Hugging Face. Please check your internet connection.")

            # Initialize the embedding model first
            self._initialize_model()
            
            # Clean and validate configuration
            cleaned_url = clean_url(settings.QDRANT_URL)
            cleaned_api_key = clean_api_key(settings.QDRANT_API_KEY)
            
            if not cleaned_url or not cleaned_api_key:
                raise ValueError("Qdrant URL and API key must not be empty")
            
            # Ensure URL starts with https://
            if not cleaned_url.startswith('https://'):
                cleaned_url = 'https://' + cleaned_url
            
            logger.info(f"Initializing Qdrant client with URL: {cleaned_url}")
            
            # Initialize Qdrant client with proper configuration
            self.client = QdrantClient(
                url=cleaned_url,
                api_key=cleaned_api_key,
                timeout=30.0,  # Increase timeout for initial connection
                prefer_grpc=False,  # Use HTTP instead of gRPC
                check_version=False,  # Disable version check
                headers={
                    "api-key": cleaned_api_key,
                    "Content-Type": "application/json"
                }
            )
            
            # Verify connection with a simple operation
            try:
                # Try to get collections as a simple connection test
                collections = self.client.get_collections()
                logger.info(f"Successfully connected to Qdrant. Found {len(collections.collections)} collections")
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {str(e)}")
                if "403" in str(e):
                    logger.error("Authentication failed. Please check your Qdrant API key and URL.")
                    logger.error(f"URL: {cleaned_url}")
                    logger.error("API Key (first 10 chars): " + cleaned_api_key[:10] + "..." if cleaned_api_key else "None")
                    # Try alternative authentication method
                    try:
                        logger.info("Trying alternative authentication method...")
                        self.client = QdrantClient(
                            url=cleaned_url,
                            api_key=cleaned_api_key,
                            timeout=30.0,
                            prefer_grpc=False,
                            check_version=False,
                            headers={
                                "Authorization": f"Bearer {cleaned_api_key}",
                                "Content-Type": "application/json"
                            }
                        )
                        collections = self.client.get_collections()
                        logger.info("Successfully connected with alternative authentication method")
                    except Exception as alt_e:
                        logger.error(f"Alternative authentication also failed: {str(alt_e)}")
                        raise
                raise
            
            # Create collection if it doesn't exist
            if not self._collection_exists():
                self._create_collection()
            
            self._initialized = True
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    def _initialize_model(self) -> None:
        """Initialize the sentence transformer model with fallback options and retry logic."""
        try:
            # Create model cache directory if it doesn't exist
            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
            # Try to load from local cache first
            if EMBEDDING_MODEL_PATH.exists() and verify_model_files(EMBEDDING_MODEL_PATH):
                logger.info(f"Loading model from local cache: {EMBEDDING_MODEL_PATH}")
                try:
                    self.model = SentenceTransformer(str(EMBEDDING_MODEL_PATH))
                    logger.info("Successfully loaded model from cache")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load model from cache: {str(e)}")
            
            # Try each model in the fallback list
            last_error = None
            for model_name in self.FALLBACK_MODELS:
                for attempt in range(self.MAX_RETRIES):
                    try:
                        logger.info(f"Attempting to load model {model_name} (attempt {attempt + 1}/{self.MAX_RETRIES})")
                        
                        # Try to download model files first
                        try:
                            logger.info(f"Downloading model files for {model_name}")
                            # Download all required files
                            for filename in ["config.json", "pytorch_model.bin", "tokenizer.json"]:
                                hf_hub_download(
                                    repo_id=f"sentence-transformers/{model_name}",
                                    filename=filename,
                                    cache_dir=str(MODEL_CACHE_DIR),
                                    token=os.getenv("HUGGINGFACE_TOKEN"),
                                    local_files_only=False,
                                    resume_download=True,
                                    force_download=False
                                )
                            logger.info(f"Successfully downloaded model files for {model_name}")
                        except Exception as e:
                            logger.warning(f"Failed to download model files: {str(e)}")
                            if "429" in str(e):
                                wait_time = self.RETRY_DELAY * (attempt + 1)
                                logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry...")
                                time.sleep(wait_time)
                            continue
                        
                        # Initialize the model
                        logger.info(f"Initializing model {model_name}")
                        self.model = SentenceTransformer(
                            model_name_or_path=model_name,
                            cache_folder=str(MODEL_CACHE_DIR)
                        )
                        
                        # Save the model locally
                        logger.info(f"Saving model {model_name} to local cache")
                        self.model.save(str(EMBEDDING_MODEL_PATH))
                        logger.info(f"Model {model_name} loaded and saved successfully")
                        return
                        
                    except HfHubHTTPError as e:
                        if "429" in str(e):
                            wait_time = self.RETRY_DELAY * (attempt + 1)
                            logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                            last_error = e
                            continue
                        raise
                    except Exception as e:
                        logger.error(f"Error loading model {model_name}: {str(e)}")
                        last_error = e
                        break
            
            if last_error:
                raise last_error
                
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def _collection_exists(self) -> bool: 
        """Check if the memory collection exists."""
        try:
            collections = self.client.get_collections().collections
            return any(col.name == self.COLLECTION_NAME for col in collections)  
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}")
            return False
    
    def _create_collection(self) -> None: 
        """Create a new collection for storing memories. """
        try:
            sample_embedding = self.model.encode("sample text")
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=len(sample_embedding),
                    distance=Distance.COSINE
                ),
            )
            logger.info(f"Created new collection: {self.COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise

    def find_similar_memory(self, text: str) -> Optional[Memory]: 
        """Find if a similar memory already exist. 
        
        Args: 
          text: The text to search for 

         Returns: 
         Optional Memory if a similar one is found 
        """ 
        try:
            results = self.search_memories(text, k=1) 
            if results and results[0].score >= self.SIMILARITY_THRESHOLD: 
                return results[0] 
            return None 
        except Exception as e:
            logger.error(f"Error finding similar memory: {str(e)}")
            return None
    
    def store_memory(self, text:str, metadata: dict) -> None: 
        """Store a new memory in the vector store or update if similar exists. 
        
        Args: 
          text: The text content of the memory 
          metadata: Additional information about the memory (timestamp, type etc)
        """
        try:
            if not self._collection_exists():
                self._create_collection() 

            #check if similar memory exists 
            similar_memory = self.find_similar_memory(text)
            if similar_memory and similar_memory.id: 
                metadata["id"] = similar_memory.id 

            embedding = self.model.encode(text) 
            point = PointStruct(
                id=metadata.get("id", hash(text)),
                vector=embedding.tolist(),
                payload={
                    "text": text,
                    **metadata,
                }
            )
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[point]
            )
            logger.info(f"Stored memory with id: {metadata.get('id')}")
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            raise
    
    def search_memories(self, query: str, k: int = 5) -> List[Memory]: 
        """Search for similar memories in a vector store 
        Args: 
          query: Text to search for 
          k: Number of results to return 

        Returns: 
           List of Memory objects
        """
        try:
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
        except Exception as e:
            logger.error(f"Error searching memories: {str(e)}")
            return []
    
@lru_cache 
def get_vector_store() -> VectorStore: 
    """Get or create the VectorStore singleton instance. """
    return VectorStore()