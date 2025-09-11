# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from .._models import BaseModel
from .file_object import FileObject

__all__ = ["Upload"]


class Upload(BaseModel):
    id: str
    """The Upload unique identifier, which can be referenced in API endpoints."""

    bytes: int
    """The intended number of bytes to be uploaded."""

    created_at: int
    """The Unix timestamp (in seconds) for when the Upload was created."""

    expires_at: int
    """The Unix timestamp (in seconds) for when the Upload will expire."""

    filename: str
    """The name of the file to be uploaded."""

    object: Literal["upload"]
    """The object type, which is always "upload"."""

    purpose: str
    """The intended purpose of the file.

    [Please refer here](https://platform.openai.com/docs/api-reference/files/object#files/object-purpose)
    for acceptable values.
    """

    status: Literal["pending", "completed", "cancelled", "expired"]
    """The status of the Upload."""

    file: Optional[FileObject] = None
    """The `File` object represents a document that has been uploaded to OpenAI."""
