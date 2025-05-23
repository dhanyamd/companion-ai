from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    GROQ_API_KEY: str
    ELEVENLABS_API_KEY: str
    ELEVENLABS_VOICE_ID: str
    TOGETHER_API_KEY: str

    QDRANT_API_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.tuHnaYj0AK3vtgxil-dMaYp8Df-ZDi8nMUuLep_Fz7I"
    QDRANT_URL: str="https://6905f4b7-7d13-42c5-ab18-b4b99584e067.eu-central-1-0.aws.cloud.qdrant.io"
    QDRANT_PORT: str = "6333"
    QDRANT_HOST: str | None = None

    TEXT_MODEL_NAME: str = "llama-3.3-70b-versatile"
    SMALL_TEXT_MODEL_NAME: str = "gemma2-9b-it"
    STT_MODEL_NAME: str = "whisper-large-v3-turbo"
    TTS_MODEL_NAME: str = "eleven_flash_v2_5"
    TTI_MODEL_NAME: str = "black-forest-labs/FLUX.1-schnell-Free"
    ITT_MODEL_NAME: str = "llama-3.2-90b-vision-preview"

    MEMORY_TOP_K: int = 3
    ROUTER_MESSAGES_TO_ANALYZE: int = 3
    TOTAL_MESSAGES_SUMMARY_TRIGGER: int = 20
    TOTAL_MESSAGES_AFTER_SUMMARY: int = 5

    SHORT_TERM_MEMORY_DB_PATH: str = "/app/data/memory.db"


settings = Settings()