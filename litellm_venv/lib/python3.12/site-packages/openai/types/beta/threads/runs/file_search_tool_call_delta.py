# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ....._models import BaseModel

__all__ = ["FileSearchToolCallDelta"]


class FileSearchToolCallDelta(BaseModel):
    file_search: object
    """For now, this is always going to be an empty object."""

    index: int
    """The index of the tool call in the tool calls array."""

    type: Literal["file_search"]
    """The type of tool call.

    This is always going to be `file_search` for this type of tool call.
    """

    id: Optional[str] = None
    """The ID of the tool call object."""
