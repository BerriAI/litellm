"""
Type definitions for ElevenLabs API responses
"""

from typing import Dict, List, Optional

from typing_extensions import TypedDict


class ElevenLabsCharacter(TypedDict, total=False):
    """Character-level timing information"""
    character: str
    start_time_seconds: float
    duration_seconds: float


class ElevenLabsAlignment(TypedDict, total=False):
    """Word-level alignment data from ElevenLabs TTS"""
    characters: List[str]
    character_start_times_seconds: List[float]
    character_end_times_seconds: List[float]


class ElevenLabsNormalizedAlignment(TypedDict, total=False):
    """Normalized alignment data with character-level timestamps"""
    characters: List[ElevenLabsCharacter]
    max_character_duration_seconds: float


class ElevenLabsTextToSpeechWithTimestampsResponse(TypedDict, total=False):
    """
    Response from ElevenLabs /v1/text-to-speech/{voice_id}/with-timestamps endpoint
    
    Reference: https://elevenlabs.io/docs/api-reference/text-to-speech-with-timestamps
    """
    audio_base_64: str  # Base64-encoded audio data
    alignment: ElevenLabsAlignment  # Word-level timing information
    normalized_alignment: ElevenLabsNormalizedAlignment  # Character-level timing information

