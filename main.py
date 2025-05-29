import uvicorn
from ai_companion.interfaces.whatsapp.webhook_endpoint import app
import os

if __name__ == "__main__":
    # Get port from environment variable (Cloud Run sets PORT=8080)
    port = int(os.getenv("PORT", 8080))
    
    # Run the FastAPI application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info"
    ) 