# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["ContainerListResponse", "ExpiresAfter"]


class ExpiresAfter(BaseModel):
    anchor: Optional[Literal["last_active_at"]] = None
    """The reference point for the expiration."""

    minutes: Optional[int] = None
    """The number of minutes after the anchor before the container expires."""


class ContainerListResponse(BaseModel):
    id: str
    """Unique identifier for the container."""

    created_at: int
    """Unix timestamp (in seconds) when the container was created."""

    name: str
    """Name of the container."""

    object: str
    """The type of this object."""

    status: str
    """Status of the container (e.g., active, deleted)."""

    expires_after: Optional[ExpiresAfter] = None
    """
    The container will expire after this time period. The anchor is the reference
    point for the expiration. The minutes is the number of minutes after the anchor
    before the container expires.
    """
