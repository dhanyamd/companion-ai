import re
import time
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from ai_companion.modules.images.image_to_text import ImageToText
from ai_companion.modules.images.text_to_image import TextToImage
from settings import settings


def get_text_to_speech_module():
    from ai_companion.speech.text_to_speech import TextToSpeech
    return TextToSpeech()


def get_text_to_image_module():
    return TextToImage()


def get_image_to_text_module():
    return ImageToText()


def remove_asterisk_content(text: str) -> str:
    """Remove content between asterisks from the text."""
    return re.sub(r"\*.*?\*", "", text).strip()


class AsteriskRemovalParser(StrOutputParser):
    def parse(self, text):
        return remove_asterisk_content(super().parse(text))