# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["EvalRunCanceledWebhookEvent", "Data"]


class Data(BaseModel):
    id: str
    """The unique ID of the eval run."""


class EvalRunCanceledWebhookEvent(BaseModel):
    id: str
    """The unique ID of the event."""

    created_at: int
    """The Unix timestamp (in seconds) of when the eval run was canceled."""

    data: Data
    """Event data payload."""

    type: Literal["eval.run.canceled"]
    """The type of the event. Always `eval.run.canceled`."""

    object: Optional[Literal["event"]] = None
    """The object of the event. Always `event`."""
