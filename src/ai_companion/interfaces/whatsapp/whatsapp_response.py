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

def sanitize_string(input_string: str) -> str:
    """Remove non-printable characters from a string, including carriage returns and newlines."""
    return re.sub(r'[^\x20-\x7E]', '', input_string)  # Keep only printable ASCII characters

# Thread pool for making synchronous requests in async context
thread_pool = ThreadPoolExecutor(max_workers=10)

def safe_http_get(url, *args, **kwargs):
    """Synchronous HTTP GET request that handles errors gracefully."""
    try:
        # Clean the URL before making the request
        if IS_CLOUD:
            url = clean_url(url)
            
        response = requests.get(url, *args, **kwargs)
        response.raise_for_status()
        return response.content
    except Exception as e:
        if IS_CLOUD:
            print(f"[CLOUD] Ignored error for URL {repr(url)}: {e}")
            print(f"[CLOUD] Error details: {traceback.format_exc()}")
            return None
        else:
            print(f"[LOCAL] Error for URL {repr(url)}: {e}")
            print(f"[LOCAL] Error details: {traceback.format_exc()}")
            raise

def safe_http_post(url, *args, **kwargs):
    """Synchronous HTTP POST request that handles errors gracefully."""
    try:
        # Clean the URL before making the request
        if IS_CLOUD:
            url = clean_url(url)
            
        response = requests.post(url, *args, **kwargs)
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return response.text
    except Exception as e:
        if IS_CLOUD:
            print(f"[CLOUD] Ignored error for URL {repr(url)}: {e}")
            print(f"[CLOUD] Error details: {traceback.format_exc()}")
            return None
        else:
            print(f"[LOCAL] Error for URL {repr(url)}: {e}")
            print(f"[LOCAL] Error details: {traceback.format_exc()}")
            raise

async def async_safe_http_get(url, *args, **kwargs):
    """Async wrapper for safe_http_get."""
    return await asyncio.get_event_loop().run_in_executor(
        thread_pool, lambda: safe_http_get(url, *args, **kwargs)
    )

async def async_safe_http_post(url, *args, **kwargs):
    """Async wrapper for safe_http_post."""
    return await asyncio.get_event_loop().run_in_executor(
        thread_pool, lambda: safe_http_post(url, *args, **kwargs)
    )

def clean_url(url: str) -> str:
    """Clean the URL by removing unwanted characters and properly encoding it."""
    try:
        # First remove any non-printable characters and trim whitespace
        # This includes \r, \n, \t and other control characters
        cleaned_url = ''.join(char for char in url if char.isprintable() or char.isspace()).strip()
        
        # Remove any remaining control characters
        cleaned_url = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned_url)
        
        # Log the URL after initial cleaning
        print(f"After initial cleaning (hex): {cleaned_url.encode('unicode_escape').decode()}")
        
        # Parse the URL to handle encoding properly
        from urllib.parse import urlparse, urlunparse, quote
        
        # Split the URL into components
        parsed = urlparse(cleaned_url)
        
        # Encode the path and query components
        encoded_path = quote(parsed.path, safe='/:')
        encoded_query = quote(parsed.query, safe='=&')
        
        # Reconstruct the URL with encoded components
        cleaned_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            encoded_path,
            parsed.params,
            encoded_query,
            parsed.fragment
        ))
        
        # Final check for any remaining non-printable characters
        cleaned_url = ''.join(char for char in cleaned_url if char.isprintable() or char.isspace()).strip()
        
        print(f"Final cleaned URL (hex): {cleaned_url.encode('unicode_escape').decode()}")
        return cleaned_url
    except Exception as e:
        print(f"Error cleaning URL: {e}")
        print(f"Error details: {traceback.format_exc()}")
        # Return a safe fallback - just the basic cleaning
        return ''.join(char for char in url if char.isprintable() or char.isspace()).strip()

class URLValidator:
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:'  # Start of group for domain/IP/localhost
            r'(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)'  # domain
            r'|localhost'  # or localhost
            r'|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'  # or IPv4
        r')'  # End of group for domain/IP/localhost
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)?$',  # path
        re.IGNORECASE
    )

    @classmethod
    def is_valid(cls, url: str) -> bool:
        if IS_CLOUD:
            return True  # Always valid in cloud
        return bool(cls.regex.match(url))

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
    # Handle POST requests for incoming messages
    try:
        # Only attempt to parse JSON for POST requests
        data = await request.json()
        print(f"Incoming data: {data}")  # Log the entire incoming data

        change_value = data.get("entry", [{}])[0].get("changes", [{}])[0].get("value", {})
        
        if "messages" in change_value:
            message = change_value.get("messages", [{}])[0]
            from_number = message.get("from", "")
            session_id = from_number

            # Handle different message types
            if message["type"] == "text":
                content = message.get("text", {}).get("body", "")
                print(f"Raw text content: {content}")

                # Sanitize the text content (do NOT use clean_url here)
                sanitized_content = sanitize_string(content)

                # Validate URLs in the text content
                urls = re.findall(r'(https?://[^\s]+)', sanitized_content)
                for url in urls:
                    if not URLValidator.is_valid(url):
                        print(f"Invalid URL found in text: {url}.")
                        return Response(content="Invalid URL in text message", status_code=400)

                print(f"Sanitized text content: {sanitized_content}")

            elif message["type"] == "audio":
                content = await process_audio_message(message)
            elif message["type"] == "image":
                if "image" in message:
                    image_id = message["image"].get("id", "")
                    if not image_id:
                        print("Image ID is missing.")
                        return Response(content="Image ID is missing", status_code=400)
                    image_bytes = await download_media(image_id)
                    # Process image...
                else:
                    print("No image data found.")
                    return Response(content="No image data found", status_code=400)
            else:
                print("Unknown message type.")
                return Response(content="Unknown message type", status_code=400)

            # Convert the message to the expected dictionary format
            message_dict = convert_message_to_dict(HumanMessage(content=sanitized_content))

            if not message_dict["content"]:
                print("Message content is empty, cannot invoke graph.")
                return Response(content="Empty message content", status_code=400)

            # Process message through the graph agent
            try:
                print(f"Attempting to connect to database at: {settings.SHORT_TERM_MEMORY_DB_PATH}")
                async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
                    print("Database connection established.")
                    graph = graph_builder.compile(checkpointer=short_term_memory)

                    # Ensure the graph is properly initialized
                    if not graph:
                        print("Graph is not properly initialized. Attempting to reinitialize.")
                        graph = graph_builder.compile(checkpointer=short_term_memory)  # Reinitialize the graph

                    if not graph:
                        print("Failed to initialize the graph.")
                        return Response(content="Graph is not properly initialized", status_code=500)

                    print("Graph compiled with checkpointer.")

                    # Sanitize URLs in the graph state if applicable
                    if hasattr(graph, 'state'):
                        for key, value in graph.state.items():
                            if isinstance(value, str) and URLValidator.is_valid(value):
                                sanitized_value = clean_url(value)
                                graph.state[key] = sanitized_value
                                print(f"Sanitized graph state value for {key}: {sanitized_value}")

                    try:
                        print(f"Invoking graph with content: {message_dict}")
                        await graph.ainvoke(
                            {"messages": [message_dict]},
                            {"configurable": {"thread_id": session_id}},
                        )
                        print("Graph invocation completed.")

                        # Get the workflow type and response from the state
                        output_state = await graph.aget_state(config={"configurable": {"thread_id": session_id}});
                        print("Retrieved output state from database.")

                        workflow = output_state.values.get("workflow", "conversation")
                        response_message = output_state.values["messages"][-1].content
                        print(response_message)

                        # Handle different response types based on workflow
                        if workflow == "audio":
                            audio_buffer = output_state.values["audio_buffer"]
                            success = await send_response(from_number, response_message, "audio", audio_buffer)
                        elif workflow == "image":
                            image_path = output_state.values["image_path"]
                            with open(image_path, "rb") as f:
                                image_data = f.read()
                            success = await send_response(from_number, response_message, "image", image_data)
                        else:
                            success = await send_response(from_number, response_message, "text")

                        if not success:
                            return Response(content="Failed to send message", status_code=500)

                        return Response(content="Message processed", status_code=200)
                    except Exception as invoke_error:
                        print(f"Error invoking graph: {invoke_error}")
                        print("Input to graph:", message_dict)
                        print("Session ID:", session_id)
                        print("Traceback:", traceback.format_exc())
                        return Response(content=f"Error invoking graph: {str(invoke_error)}", status_code=500)

                print("Exited database connection context.")

            except Exception as db_error:
                print(f"Database error: {db_error}")
                return Response(content=f"Database error: {str(db_error)}", status_code=500)

        elif "statuses" in change_value:
            return Response(content="Status update received", status_code=200)

        else:
            return Response(content="No messages found", status_code=400)

    except Exception as e:
        print(f"Error processing message: {e}")
        # Log the full traceback for debugging internal server errors
        traceback_str = traceback.format_exc()
        print("Full Traceback:", traceback_str)
        return Response(content=f"Internal server error: {str(e)}", status_code=500)

def convert_message_to_dict(message: HumanMessage) -> Dict[str, Any]:
    """Convert a HumanMessage to a dictionary format."""
    return {
        "content": message.content,
        "type": "human",  # Change this to "human" or the appropriate type
        # Add any other necessary fields here
    }

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

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')