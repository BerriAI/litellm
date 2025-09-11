# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, TypedDict

__all__ = ["RunListParams"]


class RunListParams(TypedDict, total=False):
    after: str
    """Identifier for the last run from the previous pagination request."""

    limit: int
    """Number of runs to retrieve."""

    order: Literal["asc", "desc"]
    """Sort order for runs by timestamp.

    Use `asc` for ascending order or `desc` for descending order. Defaults to `asc`.
    """

    status: Literal["queued", "in_progress", "completed", "canceled", "failed"]
    """Filter runs by status.

    One of `queued` | `in_progress` | `failed` | `completed` | `canceled`.
    """
