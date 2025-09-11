# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["InputAudioBufferClearedEvent"]


class InputAudioBufferClearedEvent(BaseModel):
    event_id: str
    """The unique ID of the server event."""

    type: Literal["input_audio_buffer.cleared"]
    """The event type, must be `input_audio_buffer.cleared`."""
