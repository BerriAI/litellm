# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

__all__ = ["ConversationItemRetrieveEventParam"]


class ConversationItemRetrieveEventParam(TypedDict, total=False):
    item_id: Required[str]
    """The ID of the item to retrieve."""

    type: Required[Literal["conversation.item.retrieve"]]
    """The event type, must be `conversation.item.retrieve`."""

    event_id: str
    """Optional client-generated ID used to identify this event."""
