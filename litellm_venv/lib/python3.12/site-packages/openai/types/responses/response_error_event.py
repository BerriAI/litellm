# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseErrorEvent"]


class ResponseErrorEvent(BaseModel):
    code: Optional[str] = None
    """The error code."""

    message: str
    """The error message."""

    param: Optional[str] = None
    """The error parameter."""

    sequence_number: int
    """The sequence number of this event."""

    type: Literal["error"]
    """The type of the event. Always `error`."""
