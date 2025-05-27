import os 
from typing import Optional 
from ai_companion.core.exceptions import TextToSpeechError 
from settings import settings, clean_api_key
from elevenlabs import generate, set_api_key, Voice, VoiceSettings
import logging

class TextToSpeech: 
    """A class to handle text-to-speech conversion using elevenlabs"""
    REQUIRED_ENV_VARS = ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]
    
    def __init__(self) -> None:
        """Initialize the text-to-speech class and validate env variables"""
        self._validate_env_vars() 
        self._initialized = False
        self.logger = logging.getLogger(__name__)

    def _validate_env_vars(self) -> None: 
        """Validate that all required env variables are set"""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars: 
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def _initialize_client(self) -> None:
        """Initialize the ElevenLabs client with API key"""
        if not self._initialized:
            try:
                # Clean the API key and ensure it's properly formatted
                cleaned_api_key = clean_api_key(settings.ELEVENLABS_API_KEY)
                if not cleaned_api_key:
                    raise ValueError("Invalid ElevenLabs API key")
                set_api_key(cleaned_api_key)
                self._initialized = True
                self.logger.info("Successfully initialized ElevenLabs client")
            except Exception as e:
                self.logger.error(f"Failed to initialize ElevenLabs client: {e}")
                raise TextToSpeechError(f"Failed to initialize ElevenLabs client: {e}")
    
    async def synthesize(self, text: str) -> bytes: 
        """Convert text to speech using Elevenlabs
        
        Args: 
            text: Text to convert to speech
        Returns: 
            bytes: Audio data
        Raises: 
            ValueError: If the input text is empty or too long 
            TextToSpeechError: If the text-to-speech conversion fails
        """
        if not text.strip(): 
            raise ValueError("Input text cannot be empty")
        if len(text) > 5000: 
            raise ValueError("Input exceeds maximum length of 5000 characters")
        
        try:
            self._initialize_client()
            self.logger.info(f"Generating speech for text: '{text[:100]}...'")
            
            # Create voice settings
            voice_settings = VoiceSettings(
                stability=0.5,
                similarity_boost=0.5
            )
            
            # Create voice object
            voice = Voice(
                voice_id=settings.ELEVENLABS_VOICE_ID,
                settings=voice_settings
            )
            
            # Generate audio
            audio = generate(
                text=text,
                voice=voice,
                model=settings.TTS_MODEL_NAME
            )
            
            if not audio:
                raise TextToSpeechError("Generated audio is empty")
                
            return audio
            
        except Exception as e:
            self.logger.error(f"Text to speech conversion failed: {str(e)}")
            raise TextToSpeechError(f"Text to speech conversion failed: {str(e)}") from e
