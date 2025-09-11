# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["InputAudioBufferSpeechStartedEvent"]


class InputAudioBufferSpeechStartedEvent(BaseModel):
    audio_start_ms: int
    """
    Milliseconds from the start of all audio written to the buffer during the
    session when speech was first detected. This will correspond to the beginning of
    audio sent to the model, and thus includes the `prefix_padding_ms` configured in
    the Session.
    """

    event_id: str
    """The unique ID of the server event."""

    item_id: str
    """The ID of the user message item that will be created when speech stops."""

    type: Literal["input_audio_buffer.speech_started"]
    """The event type, must be `input_audio_buffer.speech_started`."""
