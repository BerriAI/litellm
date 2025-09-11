# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from .._models import BaseModel
from .shared.metadata import Metadata

__all__ = ["VectorStore", "FileCounts", "ExpiresAfter"]


class FileCounts(BaseModel):
    cancelled: int
    """The number of files that were cancelled."""

    completed: int
    """The number of files that have been successfully processed."""

    failed: int
    """The number of files that have failed to process."""

    in_progress: int
    """The number of files that are currently being processed."""

    total: int
    """The total number of files."""


class ExpiresAfter(BaseModel):
    anchor: Literal["last_active_at"]
    """Anchor timestamp after which the expiration policy applies.

    Supported anchors: `last_active_at`.
    """

    days: int
    """The number of days after the anchor time that the vector store will expire."""


class VectorStore(BaseModel):
    id: str
    """The identifier, which can be referenced in API endpoints."""

    created_at: int
    """The Unix timestamp (in seconds) for when the vector store was created."""

    file_counts: FileCounts

    last_active_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the vector store was last active."""

    metadata: Optional[Metadata] = None
    """Set of 16 key-value pairs that can be attached to an object.

    This can be useful for storing additional information about the object in a
    structured format, and querying for objects via API or the dashboard.

    Keys are strings with a maximum length of 64 characters. Values are strings with
    a maximum length of 512 characters.
    """

    name: str
    """The name of the vector store."""

    object: Literal["vector_store"]
    """The object type, which is always `vector_store`."""

    status: Literal["expired", "in_progress", "completed"]
    """
    The status of the vector store, which can be either `expired`, `in_progress`, or
    `completed`. A status of `completed` indicates that the vector store is ready
    for use.
    """

    usage_bytes: int
    """The total number of bytes used by the files in the vector store."""

    expires_after: Optional[ExpiresAfter] = None
    """The expiration policy for a vector store."""

    expires_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the vector store will expire."""
