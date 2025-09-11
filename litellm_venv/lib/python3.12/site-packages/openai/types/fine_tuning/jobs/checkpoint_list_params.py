# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import TypedDict

__all__ = ["CheckpointListParams"]


class CheckpointListParams(TypedDict, total=False):
    after: str
    """Identifier for the last checkpoint ID from the previous pagination request."""

    limit: int
    """Number of checkpoints to retrieve."""
