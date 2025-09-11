# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseCompletedWebhookEvent", "Data"]


class Data(BaseModel):
    id: str
    """The unique ID of the model response."""


class ResponseCompletedWebhookEvent(BaseModel):
    id: str
    """The unique ID of the event."""

    created_at: int
    """The Unix timestamp (in seconds) of when the model response was completed."""

    data: Data
    """Event data payload."""

    type: Literal["response.completed"]
    """The type of the event. Always `response.completed`."""

    object: Optional[Literal["event"]] = None
    """The object of the event. Always `event`."""
