import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    """Download and cache the sentence transformer model."""
    # Define model paths
    MODEL_CACHE_DIR = Path("models")
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Create cache directory if it doesn't exist
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading model {EMBEDDING_MODEL}...")
    try:
        # Download and cache the model
        model = SentenceTransformer(
            EMBEDDING_MODEL,
            cache_folder=str(MODEL_CACHE_DIR)
        )
        logger.info("Model downloaded successfully!")
        
        # Save the model locally
        model_path = MODEL_CACHE_DIR / EMBEDDING_MODEL.split("/")[-1]
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        raise

if __name__ == "__main__":
    download_model() 