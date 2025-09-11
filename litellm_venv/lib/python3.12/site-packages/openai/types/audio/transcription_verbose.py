# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from typing_extensions import Literal

from ..._models import BaseModel
from .transcription_word import TranscriptionWord
from .transcription_segment import TranscriptionSegment

__all__ = ["TranscriptionVerbose", "Usage"]


class Usage(BaseModel):
    seconds: float
    """Duration of the input audio in seconds."""

    type: Literal["duration"]
    """The type of the usage object. Always `duration` for this variant."""


class TranscriptionVerbose(BaseModel):
    duration: float
    """The duration of the input audio."""

    language: str
    """The language of the input audio."""

    text: str
    """The transcribed text."""

    segments: Optional[List[TranscriptionSegment]] = None
    """Segments of the transcribed text and their corresponding details."""

    usage: Optional[Usage] = None
    """Usage statistics for models billed by audio input duration."""

    words: Optional[List[TranscriptionWord]] = None
    """Extracted words and their corresponding timestamps."""
