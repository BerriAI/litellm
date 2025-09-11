# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ...._models import BaseModel
from .message_delta import MessageDelta

__all__ = ["MessageDeltaEvent"]


class MessageDeltaEvent(BaseModel):
    id: str
    """The identifier of the message, which can be referenced in API endpoints."""

    delta: MessageDelta
    """The delta containing the fields that have changed on the Message."""

    object: Literal["thread.message.delta"]
    """The object type, which is always `thread.message.delta`."""
