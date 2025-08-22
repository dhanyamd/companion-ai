# COMPANION AI

An advanced AI companion that can chat, generate images, and process audio through WhatsApp. This project leverages multiple AI services to create an interactive and intelligent conversational experience.

<img width="738" height="1600" alt="image" src="https://github.com/user-attachments/assets/aa8b38cf-d8bc-4859-9361-9d553996c1ed" />


## 🌟 Features

- **WhatsApp Integration**: Seamless communication through WhatsApp
- **Advanced AI Capabilities**:
  - Text-to-speech and speech-to-text using ElevenLabs
  - Image generation and analysis using Together AI
  - Natural language processing with Groq
- **Memory Management**:
  - Long-term memory storage for persistent conversations
  - Short-term memory for context-aware responses
- **Conversation Management**:
  - Context-aware responses
  - Multi-turn conversation handling
  - Emotion and intent recognition

## 🛠️ Technical Stack

### AI Models & Services
- **Groq**: 
  - `llama-3.3-70b-versatile`: Main language model for conversation and reasoning
  - `gemma2-9b-it`: Smaller text model for lightweight tasks
- **Together AI**:
  - Image generation and analysis
  - `black-forest-labs/FLUX.1-schnell-Free`: Text-to-image model
  - `llama-3.2-90b-vision-preview`: Image-to-text model
- **ElevenLabs**:
  - Text-to-speech conversion
  - `eleven_flash_v2_5`: Text-to-speech model
  - Voice synthesis and audio processing
- **Whisper**:
  - `whisper-large-v3-turbo`: Speech-to-text model

### Memory Systems
- **Long-term Memory**:
  - Qdrant vector database for semantic search
  - Stores conversation history and important information
  - Enables context-aware responses across sessions
- **Short-term Memory**:
  - SQLite database for active conversation context
  - Manages current session information
  - Handles immediate context and temporary data
- **Checkpointing**:
  - DuckDB for graph state persistence
  - SQLite for async checkpointing

### Backend & Infrastructure
- **Framework**: 
  - FastAPI for high-performance API endpoints
  - LangGraph for workflow orchestration
  - LangChain for AI chain management
- **Database**: 
  - Qdrant for vector storage
  - SQLite for short-term data
  - DuckDB for checkpointing
- **Deployment**: 
  - Docker containerization
  - Google Cloud Run for serverless deployment
  - UV for Python package management
- **Monitoring**: 
  - Google Cloud Logging for error tracking
  - Pre-commit hooks for code quality
  - Ruff for linting and formatting

### Additional Tools
- **Sentence Transformers (all-MiniLM-L6-v2)**: For text embeddings
- **Supabase**: For additional data storage
- **Chainlit**: For development and testing
- **Hatchling**: For package building
- **Uvicorn**: For ASGI server

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- Docker (for containerized deployment)
- Various API keys (see Environment Variables section)

### Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [your-repo-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key

# ElevenLabs Configuration
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=your_voice_id

# Together AI Configuration
TOGETHER_API_KEY=your_together_api_key

# WhatsApp Configuration
WHATSAPP_TOKEN=your_whatsapp_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
WHATSAPP_VERIFY_TOKEN=your_verify_token

# Qdrant Configuration
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_url
```

## 📁 Project Structure

```
.
├── src/                    # Source code
│   ├── ai_companion/      # Main application code
│   │   ├── modules/       # Feature modules
│   │   │   ├── images/    # Image processing
│   │   │   ├── audio/     # Audio processing
│   │   │   └── memory/    # Memory management
│   │   └── core/         # Core functionality
├── models/                # AI model configurations
├── scripts/              # Utility scripts
├── long_term_memory/     # Qdrant vector storage
├── short_term_memory/    # SQLite database
└── ...
```

### Running Locally (quicl setup)

1. Start the development server:
```bash
make dev
```

2. For production deployment:
```bash
make prod
```

### Docker Deployment

Build and run using Docker:
```bash
docker-compose up --build
```
## 🙏 Tech

- Groq for language model capabilities
- ElevenLabs for voice synthesis
- Together AI for image generation
- WhatsApp for messaging platform
