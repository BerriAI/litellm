# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["ResponseContentPartAddedEvent", "Part"]


class Part(BaseModel):
    audio: Optional[str] = None
    """Base64-encoded audio data (if type is "audio")."""

    text: Optional[str] = None
    """The text content (if type is "text")."""

    transcript: Optional[str] = None
    """The transcript of the audio (if type is "audio")."""

    type: Optional[Literal["text", "audio"]] = None
    """The content type ("text", "audio")."""


class ResponseContentPartAddedEvent(BaseModel):
    content_index: int
    """The index of the content part in the item's content array."""

    event_id: str
    """The unique ID of the server event."""

    item_id: str
    """The ID of the item to which the content part was added."""

    output_index: int
    """The index of the output item in the response."""

    part: Part
    """The content part that was added."""

    response_id: str
    """The ID of the response."""

    type: Literal["response.content_part.added"]
    """The event type, must be `response.content_part.added`."""
