# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseAudioDeltaEvent"]


class ResponseAudioDeltaEvent(BaseModel):
    delta: str
    """A chunk of Base64 encoded response audio bytes."""

    sequence_number: int
    """A sequence number for this chunk of the stream response."""

    type: Literal["response.audio.delta"]
    """The type of the event. Always `response.audio.delta`."""
