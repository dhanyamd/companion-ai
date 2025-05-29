import os
from fastapi import FastAPI
from ai_companion.interfaces.whatsapp.whatsapp_response import whatsapp_router

# Create FastAPI app with proper configuration
app = FastAPI(
    title="AI Companion WhatsApp Webhook",
    description="Webhook endpoint for WhatsApp integration",
    version="0.1.1"
)

# Include the WhatsApp router
app.include_router(whatsapp_router)

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}