# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional

from ..._models import BaseModel

__all__ = ["FileContentResponse"]


class FileContentResponse(BaseModel):
    text: Optional[str] = None
    """The text content"""

    type: Optional[str] = None
    """The content type (currently only `"text"`)"""
