# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["ErrorEvent", "Error"]


class Error(BaseModel):
    message: str
    """A human-readable error message."""

    type: str
    """The type of error (e.g., "invalid_request_error", "server_error")."""

    code: Optional[str] = None
    """Error code, if any."""

    event_id: Optional[str] = None
    """The event_id of the client event that caused the error, if applicable."""

    param: Optional[str] = None
    """Parameter related to the error, if any."""


class ErrorEvent(BaseModel):
    error: Error
    """Details of the error."""

    event_id: str
    """The unique ID of the server event."""

    type: Literal["error"]
    """The event type, must be `error`."""
