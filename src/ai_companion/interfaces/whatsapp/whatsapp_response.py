import logging
import os
from io import BytesIO
from typing import Dict, Any
import re
import traceback
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Request, Response
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ai_companion.graph import graph_builder
from ai_companion.modules.images import ImageToText
from ai_companion.speech.speech_to_text import SpeechToText
from ai_companion.speech.text_to_speech import TextToSpeech
from ai_companion.core.utils import clean_url, sanitize_string, URLValidator
from settings import settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Global module instances
speech_to_text = SpeechToText()
text_to_speech = TextToSpeech()
image_to_text = ImageToText()
load_dotenv()

# Ensure the data directory exists
os.makedirs(os.path.dirname(settings.SHORT_TERM_MEMORY_DB_PATH), exist_ok=True)

# Router for WhatsApp response
whatsapp_router = APIRouter()

# WhatsApp API credentials
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

# Cloud environment detection
IS_CLOUD = os.getenv("RUNNING_IN_CLOUD", "0").lower() in ("1", "true", "yes")
print(f"Cloud environment detected: {IS_CLOUD} (RUNNING_IN_CLOUD={os.getenv('RUNNING_IN_CLOUD')})")

# Thread pool for making synchronous requests in async context
thread_pool = ThreadPoolExecutor(max_workers=10)

def safe_http_get(url, *args, **kwargs):
    """Synchronous HTTP GET request that handles errors gracefully."""
    try:
        response = requests.get(url, *args, **kwargs)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error for URL {repr(url)}: {e}")
        return None

def safe_http_post(url, *args, **kwargs):
    """Synchronous HTTP POST request that handles errors gracefully."""
    try:
        response = requests.post(url, *args, **kwargs)
        response.raise_for_status()
        return response.json()  # or response.text if you expect text
    except requests.exceptions.RequestException as e:
        print(f"Error for URL {repr(url)}: {e}")
        return None

async def async_safe_http_get(url, *args, **kwargs):
    """Async wrapper for safe_http_get with enhanced URL cleaning."""
    cleaned_url = clean_url(url)
    return await asyncio.get_event_loop().run_in_executor(
        thread_pool, lambda: safe_http_get(cleaned_url, *args, **kwargs)
    )

async def async_safe_http_post(url, *args, **kwargs):
    """Async wrapper for safe_http_post with enhanced URL cleaning."""
    cleaned_url = clean_url(url)
    return await asyncio.get_event_loop().run_in_executor(
        thread_pool, lambda: safe_http_post(cleaned_url, *args, **kwargs)
    )

@whatsapp_router.get("/whatsapp_response")
async def whatsapp_webhook_get(request: Request):
    # Handle GET requests for webhook verification
    params = request.query_params
    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == os.getenv("WHATSAPP_VERIFY_TOKEN")
    ):
        print("Webhook verified successfully!")
        return Response(content=params.get("hub.challenge"), status_code=200)
    print("Webhook verification failed: Invalid parameters or token")
    return Response(content="Verification failed", status_code=403)

@whatsapp_router.post("/whatsapp_response")
async def whatsapp_handler_post(request: Request):
    try:
        data = await request.json()
        print(f"Incoming data: {data}")

        change_value = data.get("entry", [{}])[0].get("changes", [{}])[0].get("value", {})
        
        if "messages" in change_value:
            message = change_value.get("messages", [{}])[0]
            from_number = message.get("from", "")
            session_id = from_number

            if message["type"] == "text":
                content = message.get("text", {}).get("body", "")
                sanitized_content = sanitize_string(content)

                # Validate URLs
                urls = re.findall(r'(https?://[^\s]+)', sanitized_content)
                for url in urls:
                    cleaned_url = clean_url(url)  # Clean the URL
                    if not URLValidator.is_valid(cleaned_url):
                        print(f"Invalid URL found: {cleaned_url}.")
                        return Response(content="Invalid URL in text message", status_code=400)

                # Log the sanitized content and URLs
                print(f"Sanitized content: {sanitized_content}")
                print(f"Valid URLs: {urls}")

            # Convert message to dict
            message_dict = convert_message_to_dict(HumanMessage(content=sanitized_content))

            # Sanitize the message content
            message_dict["content"] = sanitize_string(message_dict["content"])

            # Process message through the graph agent
            try:
                # Initialize state with required fields
                initial_state = {
                    "messages": [message_dict],
                    "summary": "",
                    "workflow": "conversation",
                    "audio_buffer": b"",
                    "image_path": "",
                    "current_activity": "",
                    "apply_activity": False,
                    "memory_context": ""
                }
                
                # Add configurable options for the graph
                config = {
                    "configurable": {
                        "thread_id": session_id,
                        "recursion_limit": 10,
                        "timeout": 30
                    },
                    "callbacks": None,
                    "tags": ["cloud_run"],
                    "metadata": {
                        "session_id": session_id,
                        "environment": "cloud_run"
                    }
                }

                # Create checkpointer with required configuration
                async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as checkpointer:
                    # Compile the graph with the checkpointer
                    graph = graph_builder.compile(checkpointer=checkpointer)
                    
                    # Try to get existing state first
                    try:
                        existing_state = await checkpointer.get({"configurable": {"thread_id": session_id}})
                        if existing_state:
                            # Merge existing state with new message
                            initial_state["messages"] = existing_state["messages"] + [message_dict]
                            initial_state["summary"] = existing_state.get("summary", "")
                            initial_state["workflow"] = existing_state.get("workflow", "conversation")
                            initial_state["current_activity"] = existing_state.get("current_activity", "")
                            initial_state["memory_context"] = existing_state.get("memory_context", "")
                    except Exception as e:
                        logger.warning(f"Could not load existing state: {e}")
                    
                    try:
                        # First try with minimal configuration
                        result = await graph.ainvoke(initial_state, {"configurable": {"thread_id": session_id}})
                        logger.info("Graph invocation completed with minimal config")
                    except Exception as inner_e:
                        logger.warning(f"Minimal config failed: {str(inner_e)}")
                        # If minimal config fails, try with full configuration
                        result = await graph.ainvoke(initial_state, config)
                        logger.info("Graph invocation completed with full config")
                    
                    if result and isinstance(result, dict):
                        print("Graph invocation completed successfully.")
                        return Response(content="Message processed successfully", status_code=200)
                    else:
                        logger.error("Graph returned invalid result")
                        return Response(content="Invalid graph result", status_code=500)
                    
            except Exception as e:
                logger.error(f"Error invoking graph: {str(e)}", exc_info=True)
                error_details = {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "message_content": sanitized_content,
                    "state": initial_state,
                    "config": config
                }
                logger.error(f"Detailed error information: {error_details}")
                return Response(content="Error processing message", status_code=500)

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        return Response(content="Internal server error", status_code=500)

def convert_message_to_dict(message: HumanMessage) -> Dict[str, Any]:
    """Convert a HumanMessage to a dictionary format."""
    result = {
        "content": message.content,
        "type": "human",  # Change this to "human" or the appropriate type
        # Add any other necessary fields here
    }
    print(f"Converted message dict: {result}")  # Log the result
    return result

async def download_media(media_id: str) -> bytes:
    sanitized_media_id = clean_url(sanitize_string(media_id))
    media_metadata_url = f"https://graph.facebook.com/v21.0/{sanitized_media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    print(f"Media metadata URL: {media_metadata_url}")
    metadata = await async_safe_http_get(media_metadata_url, headers=headers)
    if not metadata:
        print("[CLOUD] No metadata returned for media.")
        return b""
    try:
        import json
        metadata = json.loads(metadata) if isinstance(metadata, (bytes, str)) else metadata
    except Exception:
        pass
    download_url = metadata.get("url") if isinstance(metadata, dict) else None
    if download_url:
        sanitized_download_url = clean_url(download_url)
        print(f"Sanitized download URL: {sanitized_download_url}")
        media_bytes = await async_safe_http_get(sanitized_download_url, headers=headers)
        return media_bytes or b""
    else:
        print("Download URL not found in metadata.")
        return b""

async def process_audio_message(message: Dict) -> str:
    audio_id = message["audio"]["id"]
    print(f"Raw input: {audio_id}")
    sanitized_audio_id = clean_url(sanitize_string(audio_id))
    print(f"Sanitized audio ID: {sanitized_audio_id}")
    media_metadata_url = f"https://graph.facebook.com/v21.0/{sanitized_audio_id}"
    if not URLValidator.is_valid(media_metadata_url):
        print(f"Invalid URL: {media_metadata_url}")
        return ""
    print(f"Media metadata URL: {media_metadata_url}")
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    metadata = await async_safe_http_get(media_metadata_url, headers=headers)
    if not metadata:
        print("[CLOUD] No metadata returned for audio.")
        return ""
    try:
        import json
        metadata = json.loads(metadata) if isinstance(metadata, (bytes, str)) else metadata
    except Exception:
        pass
    download_url = metadata.get("url") if isinstance(metadata, dict) else None
    if not URLValidator.is_valid(download_url):
        print(f"Invalid download URL: {download_url}")
        return ""
    audio_bytes = await async_safe_http_get(download_url, headers=headers)
    if not audio_bytes:
        print("[CLOUD] No audio bytes returned.")
        return ""
    audio_buffer = BytesIO(audio_bytes)
    audio_buffer.seek(0)
    audio_data = audio_buffer.read()
    return await speech_to_text.transcribe(audio_data)

async def send_response(
    from_number: str,
    response_text: str,
    message_type: str = "text",
    media_content: bytes = None,
) -> bool:
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    if message_type in ["audio", "image"]:
        try:
            mime_type = "audio/mpeg" if message_type == "audio" else "image/png"
            media_buffer = BytesIO(media_content)
            media_id = await upload_media(media_buffer, mime_type)
            json_data = {
                "messaging_product": "whatsapp",
                "to": from_number,
                "type": message_type,
                message_type: {"id": media_id},
            }
            if message_type == "image":
                json_data["image"]["caption"] = response_text
        except Exception as e:
            logger.error(f"Media upload failed, falling back to text: {e}")
            message_type = "text"
    if message_type == "text":
        json_data = {
            "messaging_product": "whatsapp",
            "to": from_number,
            "type": "text",
            "text": {"body": response_text},
        }
    print("Sending WhatsApp response with headers:", headers)
    print("Sending WhatsApp response with data:", json_data)
    result = await async_safe_http_post(
        f"https://graph.facebook.com/v21.0/{WHATSAPP_PHONE_NUMBER_ID}/messages",
        headers=headers,
        json=json_data,
    )
    print("WhatsApp API response:", result)
    return bool(result)

async def upload_media(media_content: BytesIO, mime_type: str) -> str:
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    files = {"file": ("response.mp3", media_content, mime_type)}
    data = {"messaging_product": "whatsapp", "type": mime_type}
    result = await async_safe_http_post(
        f"https://graph.facebook.com/v21.0/{WHATSAPP_PHONE_NUMBER_ID}/media",
        headers=headers,
        data=data,
        files=files,
    )
    if not result or "id" not in result:
        print("[CLOUD] Failed to upload media or get ID.")
        return ""
    return result["id"]

def sanitize_state(state: dict) -> dict:
    """Sanitize all string values in the state dictionary."""
    sanitized = {}
    for key, value in state.items():
        if isinstance(value, str):
            sanitized[key] = sanitize_string(value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_state(value)
        elif isinstance(value, list):
            sanitized[key] = [sanitize_state(item) if isinstance(item, dict) else sanitize_string(item) if isinstance(item, str) else item for item in value]
        else:
            sanitized[key] = value
    return sanitized

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')