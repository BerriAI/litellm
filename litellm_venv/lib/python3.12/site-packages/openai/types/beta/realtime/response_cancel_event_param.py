# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

__all__ = ["ResponseCancelEventParam"]


class ResponseCancelEventParam(TypedDict, total=False):
    type: Required[Literal["response.cancel"]]
    """The event type, must be `response.cancel`."""

    event_id: str
    """Optional client-generated ID used to identify this event."""

    response_id: str
    """
    A specific response ID to cancel - if not provided, will cancel an in-progress
    response in the default conversation.
    """
