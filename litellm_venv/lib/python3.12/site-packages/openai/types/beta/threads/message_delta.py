# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from typing_extensions import Literal

from ...._models import BaseModel
from .message_content_delta import MessageContentDelta

__all__ = ["MessageDelta"]


class MessageDelta(BaseModel):
    content: Optional[List[MessageContentDelta]] = None
    """The content of the message in array of text and/or images."""

    role: Optional[Literal["user", "assistant"]] = None
    """The entity that produced the message. One of `user` or `assistant`."""
