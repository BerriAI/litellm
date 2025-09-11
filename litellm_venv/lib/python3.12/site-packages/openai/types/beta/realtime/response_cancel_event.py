# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["ResponseCancelEvent"]


class ResponseCancelEvent(BaseModel):
    type: Literal["response.cancel"]
    """The event type, must be `response.cancel`."""

    event_id: Optional[str] = None
    """Optional client-generated ID used to identify this event."""

    response_id: Optional[str] = None
    """
    A specific response ID to cancel - if not provided, will cancel an in-progress
    response in the default conversation.
    """
