import os 
import tempfile 
from typing import Optional 

from ai_companion.core.exceptions import SpeechToTextError 
from settings import settings
from groq import Groq 

class SpeechToText: 
    """A class to handle speech-to-text conversion using Groq's whisper model """
    REQUIRED_ENV_VARS = ["GROQ_API_KEY"] 

    def __init__(self) :
        """Intialise the SpeechToText class and validate environment variables. """
        self._validate_env_vars() 
        self._client: Optional[Groq] =None 

    def _validate_env_vars(self) -> None: 
        """Validate that all required environment variables are set. """
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)] 
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    @property 
    def client(self) -> Groq: 
        """Get or create Groq client instance using singleton pattern. """
        if self._client is None: 
            self._client = Groq(api_key=settings.GROQ_API_KEY)
        return self._client 
    
    async def transcribe(self, audio_data: bytes) -> str: 
        """Convert speech to text using Groq's whisper model. 
        
        Args: audio_data: Binary audio data

        Returns: 
          str: Transcribed text 

        Raises: 
           ValueError: If the audio file is empty or invalid 
           RuntimeError: If the transcription fails
        """
        if not audio_data: 
            raise ValueError("Audio data cannot be empty")
        
        try: 
            #create a temporary file with .wav extension 
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name 
            try: 
                #open the temporary file for the API request 
                with open(temp_file_path, "rb") as audio_file: 
                    transcription = self.client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-large-v3",
                        language="en",
                        response_format="text"
                    )
                if not transcription: 
                    raise SpeechToTextError("Transcription result is empty")
                return transcription 
            
            finally: 
                os.unlink(temp_file_path)
        except Exception as e: 
            raise SpeechToTextError(f"Speech-to-text conversion failed: {str(e)}")

