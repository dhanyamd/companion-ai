from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    # Required API Keys
    GROQ_API_KEY: str = Field(..., description="Groq API key for language models")
    ELEVENLABS_API_KEY: str = Field(..., description="ElevenLabs API key for text-to-speech")
    ELEVENLABS_VOICE_ID: str = Field(..., description="ElevenLabs voice ID")
    TOGETHER_API_KEY: str = Field(..., description="Together API key")

    # Qdrant Configuration
    QDRANT_API_KEY: str = Field(
        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.tuHnaYj0AK3vtgxil-dMaYp8Df-ZDi8nMUuLep_Fz7I",
        description="Qdrant API key"
    )
    QDRANT_URL: str = Field(
        default="https://6905f4b7-7d13-42c5-ab18-b4b99584e067.eu-central-1-0.aws.cloud.qdrant.io",
        description="Qdrant cloud URL"
    )
    QDRANT_PORT: str = Field(default="6333", description="Qdrant port")
    QDRANT_HOST: str | None = Field(default=None, description="Qdrant host (optional)")

    # Model Names
    TEXT_MODEL_NAME: str = Field(default="llama-3.3-70b-versatile", description="Main text model")
    SMALL_TEXT_MODEL_NAME: str = Field(default="gemma2-9b-it", description="Smaller text model")
    STT_MODEL_NAME: str = Field(default="whisper-large-v3-turbo", description="Speech-to-text model")
    TTS_MODEL_NAME: str = Field(default="eleven_flash_v2_5", description="Text-to-speech model")
    TTI_MODEL_NAME: str = Field(default="black-forest-labs/FLUX.1-schnell-Free", description="Text-to-image model")
    ITT_MODEL_NAME: str = Field(default="llama-3.2-90b-vision-preview", description="Image-to-text model")

    # Memory Settings
    MEMORY_TOP_K: int = Field(default=3, description="Number of top memories to retrieve")
    ROUTER_MESSAGES_TO_ANALYZE: int = Field(default=3, description="Number of messages to analyze for routing")
    TOTAL_MESSAGES_SUMMARY_TRIGGER: int = Field(default=20, description="Message count to trigger summary")
    TOTAL_MESSAGES_AFTER_SUMMARY: int = Field(default=5, description="Messages to keep after summary")

    # Database Path
    SHORT_TERM_MEMORY_DB_PATH: str = Field(default="/app/data/memory.db", description="Path to SQLite database")

    @validator("QDRANT_API_KEY", "QDRANT_URL")
    def clean_strings(cls, v):
        """Clean strings of whitespace and newlines."""
        if isinstance(v, str):
            # Remove all whitespace and newlines
            return ''.join(v.split())
        return v


settings = Settings()