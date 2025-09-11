# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

from .conversation_item_param import ConversationItemParam

__all__ = ["ConversationItemCreateEventParam"]


class ConversationItemCreateEventParam(TypedDict, total=False):
    item: Required[ConversationItemParam]
    """The item to add to the conversation."""

    type: Required[Literal["conversation.item.create"]]
    """The event type, must be `conversation.item.create`."""

    event_id: str
    """Optional client-generated ID used to identify this event."""

    previous_item_id: str
    """The ID of the preceding item after which the new item will be inserted.

    If not set, the new item will be appended to the end of the conversation. If set
    to `root`, the new item will be added to the beginning of the conversation. If
    set to an existing ID, it allows an item to be inserted mid-conversation. If the
    ID cannot be found, an error will be returned and the item will not be added.
    """
