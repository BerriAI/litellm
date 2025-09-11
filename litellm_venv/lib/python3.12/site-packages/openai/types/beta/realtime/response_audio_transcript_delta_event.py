# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["ResponseAudioTranscriptDeltaEvent"]


class ResponseAudioTranscriptDeltaEvent(BaseModel):
    content_index: int
    """The index of the content part in the item's content array."""

    delta: str
    """The transcript delta."""

    event_id: str
    """The unique ID of the server event."""

    item_id: str
    """The ID of the item."""

    output_index: int
    """The index of the output item in the response."""

    response_id: str
    """The ID of the response."""

    type: Literal["response.audio_transcript.delta"]
    """The event type, must be `response.audio_transcript.delta`."""
