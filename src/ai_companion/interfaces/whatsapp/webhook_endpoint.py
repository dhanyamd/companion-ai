from fastapi import FastAPI
import os
import logging
from contextlib import asynccontextmanager

from ai_companion.interfaces.whatsapp.whatsapp_response import whatsapp_router

logger = logging.getLogger(__name__)

# Get port from environment variable or default to 8080
PORT = int(os.getenv("PORT", "8080"))
HOST = os.getenv("HOST", "0.0.0.0")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up FastAPI application...")
    try:
        # Initialize any required resources here
        yield
    finally:
        # Shutdown
        logger.info("Shutting down FastAPI application...")

app = FastAPI(lifespan=lifespan)
app.include_router(whatsapp_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "webhook_endpoint:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info"
    ) 