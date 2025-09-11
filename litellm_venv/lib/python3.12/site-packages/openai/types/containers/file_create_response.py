# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["FileCreateResponse"]


class FileCreateResponse(BaseModel):
    id: str
    """Unique identifier for the file."""

    bytes: int
    """Size of the file in bytes."""

    container_id: str
    """The container this file belongs to."""

    created_at: int
    """Unix timestamp (in seconds) when the file was created."""

    object: Literal["container.file"]
    """The type of this object (`container.file`)."""

    path: str
    """Path of the file in the container."""

    source: str
    """Source of the file (e.g., `user`, `assistant`)."""
