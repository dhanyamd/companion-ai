import uvicorn
from ai_companion.interfaces.whatsapp.webhook_endpoint import app

if __name__ == "__main__":
    uvicorn.run(
        "webhook_endpoint:app",
        host="0.0.0.0",
        port=8080,
        workers=1,
        log_level="info"
    ) 