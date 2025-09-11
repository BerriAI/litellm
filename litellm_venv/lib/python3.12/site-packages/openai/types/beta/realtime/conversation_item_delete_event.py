# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["ConversationItemDeleteEvent"]


class ConversationItemDeleteEvent(BaseModel):
    item_id: str
    """The ID of the item to delete."""

    type: Literal["conversation.item.delete"]
    """The event type, must be `conversation.item.delete`."""

    event_id: Optional[str] = None
    """Optional client-generated ID used to identify this event."""
