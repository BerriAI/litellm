# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["ConversationCreatedEvent", "Conversation"]


class Conversation(BaseModel):
    id: Optional[str] = None
    """The unique ID of the conversation."""

    object: Optional[Literal["realtime.conversation"]] = None
    """The object type, must be `realtime.conversation`."""


class ConversationCreatedEvent(BaseModel):
    conversation: Conversation
    """The conversation resource."""

    event_id: str
    """The unique ID of the server event."""

    type: Literal["conversation.created"]
    """The event type, must be `conversation.created`."""
