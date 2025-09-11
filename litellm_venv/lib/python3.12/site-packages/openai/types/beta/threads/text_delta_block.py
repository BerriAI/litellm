# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ...._models import BaseModel
from .text_delta import TextDelta

__all__ = ["TextDeltaBlock"]


class TextDeltaBlock(BaseModel):
    index: int
    """The index of the content part in the message."""

    type: Literal["text"]
    """Always `text`."""

    text: Optional[TextDelta] = None
