from typing import Optional
from langchain_groq import ChatGroq
from groq import RateLimitError
from settings import settings

def create_chat_model(model_name: str, temperature: float = 0.7) -> ChatGroq:
    """Create a ChatGroq model with the specified parameters."""
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=model_name,
        temperature=temperature,
    )

def get_chat_model(temperature: float = 0.7, max_retries: int = 3, retry_delay: float = 1.0, model_name: Optional[str] = None) -> ChatGroq:
    """
    Get a chat model with rate limit handling and fallback to small model.
    
    Args:
        temperature: Model temperature
        max_retries: Maximum number of retries on rate limit
        retry_delay: Delay between retries in seconds
        model_name: Optional specific model to use
    
    Returns:
        ChatGroq: Configured chat model
    """
    # If a specific model is requested, use it
    if model_name:
        return create_chat_model(model_name, temperature)
    
    # Try with main model first
    model = create_chat_model(settings.TEXT_MODEL_NAME, temperature)
    
    # Test the model with a small request
    try:
        model.invoke("test")
        return model
    except RateLimitError:
        # If rate limited, try with small model
        return create_chat_model(settings.SMALL_TEXT_MODEL_NAME, temperature) 