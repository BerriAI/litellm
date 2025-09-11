# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["PermissionRetrieveResponse", "Data"]


class Data(BaseModel):
    id: str
    """The permission identifier, which can be referenced in the API endpoints."""

    created_at: int
    """The Unix timestamp (in seconds) for when the permission was created."""

    object: Literal["checkpoint.permission"]
    """The object type, which is always "checkpoint.permission"."""

    project_id: str
    """The project identifier that the permission is for."""


class PermissionRetrieveResponse(BaseModel):
    data: List[Data]

    has_more: bool

    object: Literal["list"]

    first_id: Optional[str] = None

    last_id: Optional[str] = None
