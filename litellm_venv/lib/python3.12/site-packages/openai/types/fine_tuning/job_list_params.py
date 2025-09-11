# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Dict, Optional
from typing_extensions import TypedDict

__all__ = ["JobListParams"]


class JobListParams(TypedDict, total=False):
    after: str
    """Identifier for the last job from the previous pagination request."""

    limit: int
    """Number of fine-tuning jobs to retrieve."""

    metadata: Optional[Dict[str, str]]
    """Optional metadata filter.

    To filter, use the syntax `metadata[k]=v`. Alternatively, set `metadata=null` to
    indicate no metadata.
    """
