"""
Type definitions for Vertex AI Text-to-Speech API

Reference: https://cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize
"""

from typing import Optional

from typing_extensions import TypedDict


class VertexTextToSpeechInput(TypedDict, total=False):
    """
    Input for Vertex AI Text-to-Speech synthesis.
    
    Exactly one of text or ssml must be provided.
    """
    text: Optional[str]
    ssml: Optional[str]


class VertexTextToSpeechVoice(TypedDict, total=False):
    """
    Voice configuration for Vertex AI Text-to-Speech.
    
    Attributes:
        languageCode: The language code (e.g., "en-US", "de-DE")
        name: The voice name (e.g., "en-US-Studio-O", "en-US-Wavenet-D")
    """
    languageCode: str
    name: str


class VertexTextToSpeechAudioConfig(TypedDict, total=False):
    """
    Audio configuration for Vertex AI Text-to-Speech.
    
    Attributes:
        audioEncoding: The audio encoding format (e.g., "LINEAR16", "MP3", "OGG_OPUS")
        speakingRate: The speaking rate (0.25 to 4.0, default "1")
    """
    audioEncoding: str
    speakingRate: str


class VertexTextToSpeechRequest(TypedDict, total=False):
    """
    Request body for Vertex AI Text-to-Speech API.
    
    Reference: https://cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize
    """
    input: VertexTextToSpeechInput
    voice: VertexTextToSpeechVoice
    audioConfig: Optional[VertexTextToSpeechAudioConfig]
