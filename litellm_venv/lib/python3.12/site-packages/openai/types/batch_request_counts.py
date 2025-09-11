# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from .._models import BaseModel

__all__ = ["BatchRequestCounts"]


class BatchRequestCounts(BaseModel):
    completed: int
    """Number of requests that have been completed successfully."""

    failed: int
    """Number of requests that have failed."""

    total: int
    """Total number of requests in the batch."""
