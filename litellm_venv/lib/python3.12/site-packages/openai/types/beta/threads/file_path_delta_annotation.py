# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["FilePathDeltaAnnotation", "FilePath"]


class FilePath(BaseModel):
    file_id: Optional[str] = None
    """The ID of the file that was generated."""


class FilePathDeltaAnnotation(BaseModel):
    index: int
    """The index of the annotation in the text content part."""

    type: Literal["file_path"]
    """Always `file_path`."""

    end_index: Optional[int] = None

    file_path: Optional[FilePath] = None

    start_index: Optional[int] = None

    text: Optional[str] = None
    """The text in the message content that needs to be replaced."""
