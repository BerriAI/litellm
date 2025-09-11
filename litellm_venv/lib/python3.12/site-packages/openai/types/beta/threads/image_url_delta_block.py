# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ...._models import BaseModel
from .image_url_delta import ImageURLDelta

__all__ = ["ImageURLDeltaBlock"]


class ImageURLDeltaBlock(BaseModel):
    index: int
    """The index of the content part in the message."""

    type: Literal["image_url"]
    """Always `image_url`."""

    image_url: Optional[ImageURLDelta] = None
