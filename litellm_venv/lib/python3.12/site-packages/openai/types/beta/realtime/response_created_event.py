# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ...._models import BaseModel
from .realtime_response import RealtimeResponse

__all__ = ["ResponseCreatedEvent"]


class ResponseCreatedEvent(BaseModel):
    event_id: str
    """The unique ID of the server event."""

    response: RealtimeResponse
    """The response resource."""

    type: Literal["response.created"]
    """The event type, must be `response.created`."""
