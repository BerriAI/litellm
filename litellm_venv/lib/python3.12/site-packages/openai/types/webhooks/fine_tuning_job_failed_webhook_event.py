# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["FineTuningJobFailedWebhookEvent", "Data"]


class Data(BaseModel):
    id: str
    """The unique ID of the fine-tuning job."""


class FineTuningJobFailedWebhookEvent(BaseModel):
    id: str
    """The unique ID of the event."""

    created_at: int
    """The Unix timestamp (in seconds) of when the fine-tuning job failed."""

    data: Data
    """Event data payload."""

    type: Literal["fine_tuning.job.failed"]
    """The type of the event. Always `fine_tuning.job.failed`."""

    object: Optional[Literal["event"]] = None
    """The object of the event. Always `event`."""
