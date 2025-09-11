# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["ConversationItemTruncateEvent"]


class ConversationItemTruncateEvent(BaseModel):
    audio_end_ms: int
    """Inclusive duration up to which audio is truncated, in milliseconds.

    If the audio_end_ms is greater than the actual audio duration, the server will
    respond with an error.
    """

    content_index: int
    """The index of the content part to truncate. Set this to 0."""

    item_id: str
    """The ID of the assistant message item to truncate.

    Only assistant message items can be truncated.
    """

    type: Literal["conversation.item.truncate"]
    """The event type, must be `conversation.item.truncate`."""

    event_id: Optional[str] = None
    """Optional client-generated ID used to identify this event."""
