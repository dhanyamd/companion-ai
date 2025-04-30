import os 
from typing import Optional 
from ai_companion.core.exceptions import TextToSpeechError 
from settings import settings 
from elevenlabs import ElevenLabs, Voice, VoiceSettings

class TextToSpeech: 
    """A class to handle text-to-speech conversion using elevenlabs"""
    REQUIRED_ENV_VARS = ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]
    
    def __init__(self) -> None:
        """Intialize the text-to-speech class and validate env variables"""
        self._validate_env_vars() 
        self._client: Optional[ElevenLabs] =None 

    def _validate_env_vars(self) -> None: 
        """Validate that all requiered env variables are set"""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars: 
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    @property 
    def client(self) -> ElevenLabs: 
        """Get or create Elevenlabs client instance using singleton pattern. """
        if self._client is None: 
            self._client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
        return self._client 
    
    async def synthesize(self, text:str) -> bytes: 
        """Convert text to speech using Elevenlabs
        Args: 
           text: Text to convert to speech
        Returns: 
           bytes: Audio data
        Raises: 
           ValueError: If the input text is empty or too long 
           TextToSpeecgError: If the text-to-speech conversion fails
        """
        if not text.strip(): 
            raise ValueError("Input text cannot be empty")
        if len(text) > 5000: 
            raise("Input exceeds maximum length of 5000 characters")
        
        try: 
            audio_generator = self.client.generate(
                text=text,
                voice=Voice(
                    voice_id=settings.ELEVENLABS_VOICE_ID,
                    settings=VoiceSettings(stability=0.5, similarity_boost=0.5)
                ),
                model=settings.TTS_MODEL_NAME
            )

            #convert generator to bytes 
            audio_bytes = b"".join(audio_generator)
            if not audio_bytes: 
                raise TextToSpeechError("Generated audio is empty")
            return audio_bytes 
        except Exception as e: 
            raise TextToSpeechError(f"Text to speech conversion failes: {str(e)}") from e
