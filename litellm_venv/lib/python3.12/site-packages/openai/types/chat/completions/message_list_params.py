# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, TypedDict

__all__ = ["MessageListParams"]


class MessageListParams(TypedDict, total=False):
    after: str
    """Identifier for the last message from the previous pagination request."""

    limit: int
    """Number of messages to retrieve."""

    order: Literal["asc", "desc"]
    """Sort order for messages by timestamp.

    Use `asc` for ascending order or `desc` for descending order. Defaults to `asc`.
    """
