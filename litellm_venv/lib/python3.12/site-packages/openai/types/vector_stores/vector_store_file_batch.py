# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["VectorStoreFileBatch", "FileCounts"]


class FileCounts(BaseModel):
    cancelled: int
    """The number of files that where cancelled."""

    completed: int
    """The number of files that have been processed."""

    failed: int
    """The number of files that have failed to process."""

    in_progress: int
    """The number of files that are currently being processed."""

    total: int
    """The total number of files."""


class VectorStoreFileBatch(BaseModel):
    id: str
    """The identifier, which can be referenced in API endpoints."""

    created_at: int
    """
    The Unix timestamp (in seconds) for when the vector store files batch was
    created.
    """

    file_counts: FileCounts

    object: Literal["vector_store.files_batch"]
    """The object type, which is always `vector_store.file_batch`."""

    status: Literal["in_progress", "completed", "cancelled", "failed"]
    """
    The status of the vector store files batch, which can be either `in_progress`,
    `completed`, `cancelled` or `failed`.
    """

    vector_store_id: str
    """
    The ID of the
    [vector store](https://platform.openai.com/docs/api-reference/vector-stores/object)
    that the [File](https://platform.openai.com/docs/api-reference/files) is
    attached to.
    """
