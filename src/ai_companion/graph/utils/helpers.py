import re
import time
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from groq import RateLimitError

from ai_companion.modules.images.image_to_text import ImageToText
from ai_companion.modules.images.text_to_image import TextToImage
from settings import settings


def get_chat_model(temperature: float = 0.7, max_retries: int = 3, retry_delay: float = 1.0) -> ChatGroq:
    """
    Get a chat model with rate limit handling and fallback to small model.
    
    Args:
        temperature: Model temperature
        max_retries: Maximum number of retries on rate limit
        retry_delay: Delay between retries in seconds
    
    Returns:
        ChatGroq: Configured chat model
    """
    def create_model(model_name: str) -> ChatGroq:
        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name=model_name,
            temperature=temperature,
        )
    
    # Try with main model first
    model = create_model(settings.TEXT_MODEL_NAME)
    
    # Test the model with a small request
    try:
        model.invoke("test")
        return model
    except RateLimitError:
        # If rate limited, try with small model
        return create_model(settings.SMALL_TEXT_MODEL_NAME)


def get_text_to_speech_module():
    from ai_companion.speech.text_to_speech import TextToSpeech
    return TextToSpeech()


def get_text_to_image_module():
    return TextToImage()


def get_image_to_text_module():
    return ImageToText()


def remove_asterisk_content(text: str) -> str:
    """Remove content between asterisks from the text."""
    return re.sub(r"\*.*?\*", "", text).strip()


class AsteriskRemovalParser(StrOutputParser):
    def parse(self, text):
        return remove_asterisk_content(super().parse(text))