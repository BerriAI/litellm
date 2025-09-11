# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["InputAudioBufferSpeechStoppedEvent"]


class InputAudioBufferSpeechStoppedEvent(BaseModel):
    audio_end_ms: int
    """Milliseconds since the session started when speech stopped.

    This will correspond to the end of audio sent to the model, and thus includes
    the `min_silence_duration_ms` configured in the Session.
    """

    event_id: str
    """The unique ID of the server event."""

    item_id: str
    """The ID of the user message item that will be created."""

    type: Literal["input_audio_buffer.speech_stopped"]
    """The event type, must be `input_audio_buffer.speech_stopped`."""
