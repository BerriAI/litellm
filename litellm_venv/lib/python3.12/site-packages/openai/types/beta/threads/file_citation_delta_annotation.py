# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["FileCitationDeltaAnnotation", "FileCitation"]


class FileCitation(BaseModel):
    file_id: Optional[str] = None
    """The ID of the specific File the citation is from."""

    quote: Optional[str] = None
    """The specific quote in the file."""


class FileCitationDeltaAnnotation(BaseModel):
    index: int
    """The index of the annotation in the text content part."""

    type: Literal["file_citation"]
    """Always `file_citation`."""

    end_index: Optional[int] = None

    file_citation: Optional[FileCitation] = None

    start_index: Optional[int] = None

    text: Optional[str] = None
    """The text in the message content that needs to be replaced."""
