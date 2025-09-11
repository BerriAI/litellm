# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["UploadPart"]


class UploadPart(BaseModel):
    id: str
    """The upload Part unique identifier, which can be referenced in API endpoints."""

    created_at: int
    """The Unix timestamp (in seconds) for when the Part was created."""

    object: Literal["upload.part"]
    """The object type, which is always `upload.part`."""

    upload_id: str
    """The ID of the Upload object that this Part was added to."""
