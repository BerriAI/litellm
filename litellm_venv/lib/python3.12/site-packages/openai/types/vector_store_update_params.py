# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Literal, Required, TypedDict

from .shared_params.metadata import Metadata

__all__ = ["VectorStoreUpdateParams", "ExpiresAfter"]


class VectorStoreUpdateParams(TypedDict, total=False):
    expires_after: Optional[ExpiresAfter]
    """The expiration policy for a vector store."""

    metadata: Optional[Metadata]
    """Set of 16 key-value pairs that can be attached to an object.

    This can be useful for storing additional information about the object in a
    structured format, and querying for objects via API or the dashboard.

    Keys are strings with a maximum length of 64 characters. Values are strings with
    a maximum length of 512 characters.
    """

    name: Optional[str]
    """The name of the vector store."""


class ExpiresAfter(TypedDict, total=False):
    anchor: Required[Literal["last_active_at"]]
    """Anchor timestamp after which the expiration policy applies.

    Supported anchors: `last_active_at`.
    """

    days: Required[int]
    """The number of days after the anchor time that the vector store will expire."""
