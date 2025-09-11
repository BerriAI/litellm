# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["ConversationItemTruncatedEvent"]


class ConversationItemTruncatedEvent(BaseModel):
    audio_end_ms: int
    """The duration up to which the audio was truncated, in milliseconds."""

    content_index: int
    """The index of the content part that was truncated."""

    event_id: str
    """The unique ID of the server event."""

    item_id: str
    """The ID of the assistant message item that was truncated."""

    type: Literal["conversation.item.truncated"]
    """The event type, must be `conversation.item.truncated`."""
