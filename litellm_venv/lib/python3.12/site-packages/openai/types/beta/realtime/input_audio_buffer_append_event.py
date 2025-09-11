# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["InputAudioBufferAppendEvent"]


class InputAudioBufferAppendEvent(BaseModel):
    audio: str
    """Base64-encoded audio bytes.

    This must be in the format specified by the `input_audio_format` field in the
    session configuration.
    """

    type: Literal["input_audio_buffer.append"]
    """The event type, must be `input_audio_buffer.append`."""

    event_id: Optional[str] = None
    """Optional client-generated ID used to identify this event."""
