# AI Companion

An AI companion that can chat, generate images, and process audio through WhatsApp.

## Features

- WhatsApp integration
- Text-to-speech and speech-to-text capabilities
- Image generation and analysis
- Long-term memory storage
- Conversation management

## Requirements

- Python 3.12 or higher
- Various API keys (Groq, ElevenLabs, Together AI, etc.)

## Installation

```bash
pip install -e .
```

## Environment Variables

The following environment variables are required:

- `GROQ_API_KEY`: API key for Groq language models
- `ELEVENLABS_API_KEY`: API key for ElevenLabs text-to-speech
- `ELEVENLABS_VOICE_ID`: Voice ID for ElevenLabs
- `TOGETHER_API_KEY`: API key for Together AI
- `WHATSAPP_TOKEN`: WhatsApp API token
- `WHATSAPP_PHONE_NUMBER_ID`: WhatsApp phone number ID
- `WHATSAPP_VERIFY_TOKEN`: WhatsApp webhook verification token
