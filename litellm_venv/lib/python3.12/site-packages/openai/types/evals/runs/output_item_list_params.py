# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

__all__ = ["OutputItemListParams"]


class OutputItemListParams(TypedDict, total=False):
    eval_id: Required[str]

    after: str
    """Identifier for the last output item from the previous pagination request."""

    limit: int
    """Number of output items to retrieve."""

    order: Literal["asc", "desc"]
    """Sort order for output items by timestamp.

    Use `asc` for ascending order or `desc` for descending order. Defaults to `asc`.
    """

    status: Literal["fail", "pass"]
    """Filter output items by status.

    Use `failed` to filter by failed output items or `pass` to filter by passed
    output items.
    """
