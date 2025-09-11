# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Dict, List, Union, Optional
from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["VectorStoreSearchResponse", "Content"]


class Content(BaseModel):
    text: str
    """The text content returned from search."""

    type: Literal["text"]
    """The type of content."""


class VectorStoreSearchResponse(BaseModel):
    attributes: Optional[Dict[str, Union[str, float, bool]]] = None
    """Set of 16 key-value pairs that can be attached to an object.

    This can be useful for storing additional information about the object in a
    structured format, and querying for objects via API or the dashboard. Keys are
    strings with a maximum length of 64 characters. Values are strings with a
    maximum length of 512 characters, booleans, or numbers.
    """

    content: List[Content]
    """Content chunks from the file."""

    file_id: str
    """The ID of the vector store file."""

    filename: str
    """The name of the vector store file."""

    score: float
    """The similarity score for the result."""
