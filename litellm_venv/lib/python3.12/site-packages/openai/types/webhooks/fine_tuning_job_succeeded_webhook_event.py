# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["FineTuningJobSucceededWebhookEvent", "Data"]


class Data(BaseModel):
    id: str
    """The unique ID of the fine-tuning job."""


class FineTuningJobSucceededWebhookEvent(BaseModel):
    id: str
    """The unique ID of the event."""

    created_at: int
    """The Unix timestamp (in seconds) of when the fine-tuning job succeeded."""

    data: Data
    """Event data payload."""

    type: Literal["fine_tuning.job.succeeded"]
    """The type of the event. Always `fine_tuning.job.succeeded`."""

    object: Optional[Literal["event"]] = None
    """The object of the event. Always `event`."""
