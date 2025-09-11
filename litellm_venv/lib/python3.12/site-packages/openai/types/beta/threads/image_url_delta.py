# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["ImageURLDelta"]


class ImageURLDelta(BaseModel):
    detail: Optional[Literal["auto", "low", "high"]] = None
    """Specifies the detail level of the image.

    `low` uses fewer tokens, you can opt in to high resolution using `high`.
    """

    url: Optional[str] = None
    """
    The URL of the image, must be a supported image types: jpeg, jpg, png, gif,
    webp.
    """
