# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseAudioTranscriptDoneEvent"]


class ResponseAudioTranscriptDoneEvent(BaseModel):
    sequence_number: int
    """The sequence number of this event."""

    type: Literal["response.audio.transcript.done"]
    """The type of the event. Always `response.audio.transcript.done`."""
