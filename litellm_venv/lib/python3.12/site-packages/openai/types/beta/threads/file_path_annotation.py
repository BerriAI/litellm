# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["FilePathAnnotation", "FilePath"]


class FilePath(BaseModel):
    file_id: str
    """The ID of the file that was generated."""


class FilePathAnnotation(BaseModel):
    end_index: int

    file_path: FilePath

    start_index: int

    text: str
    """The text in the message content that needs to be replaced."""

    type: Literal["file_path"]
    """Always `file_path`."""
