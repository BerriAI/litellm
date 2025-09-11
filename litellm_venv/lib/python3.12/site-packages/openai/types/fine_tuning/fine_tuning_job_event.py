# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

import builtins
from typing import Optional
from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["FineTuningJobEvent"]


class FineTuningJobEvent(BaseModel):
    id: str
    """The object identifier."""

    created_at: int
    """The Unix timestamp (in seconds) for when the fine-tuning job was created."""

    level: Literal["info", "warn", "error"]
    """The log level of the event."""

    message: str
    """The message of the event."""

    object: Literal["fine_tuning.job.event"]
    """The object type, which is always "fine_tuning.job.event"."""

    data: Optional[builtins.object] = None
    """The data associated with the event."""

    type: Optional[Literal["message", "metrics"]] = None
    """The type of event."""
