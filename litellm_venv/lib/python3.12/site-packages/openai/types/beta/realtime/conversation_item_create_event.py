# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ...._models import BaseModel
from .conversation_item import ConversationItem

__all__ = ["ConversationItemCreateEvent"]


class ConversationItemCreateEvent(BaseModel):
    item: ConversationItem
    """The item to add to the conversation."""

    type: Literal["conversation.item.create"]
    """The event type, must be `conversation.item.create`."""

    event_id: Optional[str] = None
    """Optional client-generated ID used to identify this event."""

    previous_item_id: Optional[str] = None
    """The ID of the preceding item after which the new item will be inserted.

    If not set, the new item will be appended to the end of the conversation. If set
    to `root`, the new item will be added to the beginning of the conversation. If
    set to an existing ID, it allows an item to be inserted mid-conversation. If the
    ID cannot be found, an error will be returned and the item will not be added.
    """
