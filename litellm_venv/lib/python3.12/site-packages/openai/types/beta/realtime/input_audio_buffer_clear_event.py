# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["InputAudioBufferClearEvent"]


class InputAudioBufferClearEvent(BaseModel):
    type: Literal["input_audio_buffer.clear"]
    """The event type, must be `input_audio_buffer.clear`."""

    event_id: Optional[str] = None
    """Optional client-generated ID used to identify this event."""
