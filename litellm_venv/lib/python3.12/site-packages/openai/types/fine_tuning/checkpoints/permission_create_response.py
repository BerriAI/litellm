# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["PermissionCreateResponse"]


class PermissionCreateResponse(BaseModel):
    id: str
    """The permission identifier, which can be referenced in the API endpoints."""

    created_at: int
    """The Unix timestamp (in seconds) for when the permission was created."""

    object: Literal["checkpoint.permission"]
    """The object type, which is always "checkpoint.permission"."""

    project_id: str
    """The project identifier that the permission is for."""
