# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, TypedDict

__all__ = ["EvalListParams"]


class EvalListParams(TypedDict, total=False):
    after: str
    """Identifier for the last eval from the previous pagination request."""

    limit: int
    """Number of evals to retrieve."""

    order: Literal["asc", "desc"]
    """Sort order for evals by timestamp.

    Use `asc` for ascending order or `desc` for descending order.
    """

    order_by: Literal["created_at", "updated_at"]
    """Evals can be ordered by creation time or last updated time.

    Use `created_at` for creation time or `updated_at` for last updated time.
    """
