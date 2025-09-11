# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["BatchCompletedWebhookEvent", "Data"]


class Data(BaseModel):
    id: str
    """The unique ID of the batch API request."""


class BatchCompletedWebhookEvent(BaseModel):
    id: str
    """The unique ID of the event."""

    created_at: int
    """The Unix timestamp (in seconds) of when the batch API request was completed."""

    data: Data
    """Event data payload."""

    type: Literal["batch.completed"]
    """The type of the event. Always `batch.completed`."""

    object: Optional[Literal["event"]] = None
    """The object of the event. Always `event`."""
