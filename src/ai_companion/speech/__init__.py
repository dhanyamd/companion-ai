def get_speech_to_text():
    from ai_companion.speech.speech_to_text import SpeechToText
    return SpeechToText

def get_text_to_speech():
    from ai_companion.speech.text_to_speech import TextToSpeech
    return TextToSpeech

__all__ = ["get_speech_to_text", "get_text_to_speech"]