import os
import logging
from typing import Optional
from elevenlabs.client import Client
from elevenlabs import Voice, VoiceSettings

logger = logging.getLogger(__name__)

class TextToSpeechError(Exception):
    """Custom exception for text-to-speech errors."""
    pass

class TextToSpeech:
    """Handles text-to-speech conversion using ElevenLabs."""
    
    def __init__(self):
        """Initialize the text-to-speech handler."""
        self._validate_env()
        self._initialize_client()
        
    def _validate_env(self) -> None:
        """Validate required environment variables."""
        if not os.getenv("ELEVENLABS_API_KEY"):
            raise TextToSpeechError("ELEVENLABS_API_KEY environment variable is not set")
            
    def _initialize_client(self) -> None:
        """Initialize the ElevenLabs client."""
        try:
            self.client = Client(api_key=os.getenv("ELEVENLABS_API_KEY"))
        except Exception as e:
            raise TextToSpeechError(f"Failed to initialize ElevenLabs client: {str(e)}")
            
    async def synthesize(self, text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> Optional[bytes]:
        """
        Synthesize speech from text.
        
        Args:
            text: The text to synthesize
            voice_id: The ID of the voice to use
            
        Returns:
            bytes: The synthesized audio data, or None if synthesis fails
        """
        if not text:
            raise TextToSpeechError("No text provided for synthesis")
            
        try:
            # Create voice settings
            voice_settings = VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True
            )
            
            # Create voice object
            voice = Voice(
                voice_id=voice_id,
                settings=voice_settings
            )
            
            # Generate audio using the client
            audio = self.client.generate(
                text=text,
                voice=voice,
                model="eleven_multilingual_v2"
            )
            
            # Convert audio to bytes if it's not already
            if isinstance(audio, str):
                with open(audio, 'rb') as f:
                    audio_bytes = f.read()
                return audio_bytes
            elif isinstance(audio, bytes):
                return audio
            else:
                raise TextToSpeechError(f"Unexpected audio type: {type(audio)}")
            
        except Exception as e:
            logger.error(f"Text-to-speech synthesis failed: {str(e)}")
            raise TextToSpeechError(f"Failed to synthesize speech: {str(e)}")
