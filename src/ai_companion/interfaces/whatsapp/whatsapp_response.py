import logging
import os
from io import BytesIO
from typing import Dict, Any
import re
import traceback
import httpx
from fastapi import APIRouter, Request, Response
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ai_companion.graph import graph_builder
from ai_companion.modules.images import ImageToText
from ai_companion.core.utils import clean_url, sanitize_string, URLValidator
from ai_companion.graph.state import AICompanionState
from settings import settings
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Router for WhatsApp response
whatsapp_router = APIRouter()

# WhatsApp API credentials
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "").strip()
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip()

# Cloud environment detection
IS_CLOUD = os.getenv("RUNNING_IN_CLOUD", "0").lower() in ("1", "true", "yes")
print(f"Cloud environment detected: {IS_CLOUD} (RUNNING_IN_CLOUD={os.getenv('RUNNING_IN_CLOUD')})")

# Lazy-loaded components
_speech_to_text = None
_text_to_speech = None
_image_to_text = None

def get_speech_to_text():
    global _speech_to_text
    if _speech_to_text is None:
        from ai_companion.speech.speech_to_text import SpeechToText
        _speech_to_text = SpeechToText()
    return _speech_to_text

def get_text_to_speech():
    global _text_to_speech
    if _text_to_speech is None:
        from ai_companion.speech.text_to_speech import TextToSpeech
        _text_to_speech = TextToSpeech()
    return _text_to_speech

def get_image_to_text():
    global _image_to_text
    if _image_to_text is None:
        _image_to_text = ImageToText()
    return _image_to_text

# Ensure the data directory exists
os.makedirs(os.path.dirname(settings.SHORT_TERM_MEMORY_DB_PATH), exist_ok=True)

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

            # Get user message and handle different message types
            content = ""
            if message["type"] == "audio":
                content = await process_audio_message(message)
            elif message["type"] == "image":
                # Get image caption if any
                content = message.get("image", {}).get("caption", "")
                # Download and analyze image
                image_bytes = await download_media(message["image"]["id"])
                try:
                    description = await get_image_to_text().analyze_image(
                        image_bytes,
                        "Please describe what you see in this image in the context of our conversation.",
                    )
                    content += f"\n[Image Analysis: {description}]"
                except Exception as e:
                    logger.warning(f"Failed to analyze image: {e}")
            else:
                content = message["text"]["body"]

            # Sanitize the content
            content = sanitize_string(content)

            # Process message through the graph agent
            async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as checkpointer:
                graph = graph_builder.compile(checkpointer=checkpointer)
                
                # Create initial state with AICompanionState
                initial_state = AICompanionState(
                    messages=[HumanMessage(content=content)],
                    summary="",
                    workflow="conversation",
                    audio_buffer=b"",
                    image_path="",
                    current_activity="",
                    apply_activity=False,
                    memory_context=""
                )
                
                await graph.ainvoke(
                    initial_state,
                    {"configurable": {"thread_id": session_id}},
                )

                # Get the workflow type and response from the state
                output_state = await graph.aget_state(config={"configurable": {"thread_id": session_id}})

            workflow = output_state.values.get("workflow", "conversation")
            response_message = output_state.values["messages"][-1].content

            # Clean the response message
            response_message = sanitize_string(response_message)

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

        elif "statuses" in change_value:
            return Response(content="Status update received", status_code=200)

        else:
            return Response(content="Unknown event type", status_code=400)

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        return Response(content="Internal server error", status_code=500)

async def download_media(media_id: str) -> bytes:
    """Download media from WhatsApp."""
    media_metadata_url = f"https://graph.facebook.com/v21.0/{media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

    async with httpx.AsyncClient() as client:
        metadata_response = await client.get(media_metadata_url, headers=headers)
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        download_url = metadata.get("url")

        media_response = await client.get(download_url, headers=headers)
        media_response.raise_for_status()
        return media_response.content

async def process_audio_message(message: Dict[str, Any]) -> str:
    """Process an audio message by downloading and transcribing it."""
    try:
        # Get the media ID
        media_id = message["audio"]["id"]
        
        # Get the download URL
        headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://graph.facebook.com/v21.0/{media_id}",
                headers=headers
            )
            response.raise_for_status()
            download_url = response.json()["url"]

        # Download the audio file
        async with httpx.AsyncClient() as client:
            audio_response = await client.get(download_url, headers=headers)
            audio_response.raise_for_status()

        # Prepare for transcription
        audio_buffer = BytesIO(audio_response.content)
        audio_buffer.seek(0)
        audio_data = audio_buffer.read()

        return await get_speech_to_text().transcribe(audio_data)

    except Exception as e:
        logger.error(f"Error processing audio message: {e}")
        raise

async def send_response(
    from_number: str,
    response_text: str,
    message_type: str = "text",
    media_content: bytes = None,
) -> bool:
    """Send response to user via WhatsApp API."""
    # Clean the token to remove any whitespace or newlines
    clean_token = WHATSAPP_TOKEN.strip()
    headers = {
        "Authorization": f"Bearer {clean_token}",
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

            # Add caption for images
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

    print(headers)
    print(json_data)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://graph.facebook.com/v21.0/{WHATSAPP_PHONE_NUMBER_ID}/messages",
            headers=headers,
            json=json_data,
        )

    return response.status_code == 200

async def upload_media(media_content: BytesIO, mime_type: str) -> str:
    """Upload media to WhatsApp servers."""
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    files = {"file": ("response.mp3", media_content, mime_type)}
    data = {"messaging_product": "whatsapp", "type": mime_type}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://graph.facebook.com/v21.0/{WHATSAPP_PHONE_NUMBER_ID}/media",
            headers=headers,
            files=files,
            data=data,
        )
        result = response.json()

    if "id" not in result:
        raise Exception("Failed to upload media")
    return result["id"]