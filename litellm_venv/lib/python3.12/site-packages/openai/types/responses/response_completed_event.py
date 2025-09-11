# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from .response import Response
from ..._models import BaseModel

__all__ = ["ResponseCompletedEvent"]


class ResponseCompletedEvent(BaseModel):
    response: Response
    """Properties of the completed response."""

    sequence_number: int
    """The sequence number for this event."""

    type: Literal["response.completed"]
    """The type of the event. Always `response.completed`."""
