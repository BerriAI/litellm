# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["RealtimeResponseStatus", "Error"]


class Error(BaseModel):
    code: Optional[str] = None
    """Error code, if any."""

    type: Optional[str] = None
    """The type of error."""


class RealtimeResponseStatus(BaseModel):
    error: Optional[Error] = None
    """
    A description of the error that caused the response to fail, populated when the
    `status` is `failed`.
    """

    reason: Optional[
        Literal[
            "turn_detected", "client_cancelled", "max_output_tokens", "content_filter"
        ]
    ] = None
    """The reason the Response did not complete.

    For a `cancelled` Response, one of `turn_detected` (the server VAD detected a
    new start of speech) or `client_cancelled` (the client sent a cancel event). For
    an `incomplete` Response, one of `max_output_tokens` or `content_filter` (the
    server-side safety filter activated and cut off the response).
    """

    type: Optional[Literal["completed", "cancelled", "incomplete", "failed"]] = None
    """
    The type of error that caused the response to fail, corresponding with the
    `status` field (`completed`, `cancelled`, `incomplete`, `failed`).
    """
