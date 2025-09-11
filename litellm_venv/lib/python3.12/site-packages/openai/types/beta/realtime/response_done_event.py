# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ...._models import BaseModel
from .realtime_response import RealtimeResponse

__all__ = ["ResponseDoneEvent"]


class ResponseDoneEvent(BaseModel):
    event_id: str
    """The unique ID of the server event."""

    response: RealtimeResponse
    """The response resource."""

    type: Literal["response.done"]
    """The event type, must be `response.done`."""
