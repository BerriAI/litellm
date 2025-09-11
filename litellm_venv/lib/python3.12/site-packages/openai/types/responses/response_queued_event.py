# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from .response import Response
from ..._models import BaseModel

__all__ = ["ResponseQueuedEvent"]


class ResponseQueuedEvent(BaseModel):
    response: Response
    """The full response object that is queued."""

    sequence_number: int
    """The sequence number for this event."""

    type: Literal["response.queued"]
    """The type of the event. Always 'response.queued'."""
