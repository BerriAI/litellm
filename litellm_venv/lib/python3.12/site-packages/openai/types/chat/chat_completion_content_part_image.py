# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ChatCompletionContentPartImage", "ImageURL"]


class ImageURL(BaseModel):
    url: str
    """Either a URL of the image or the base64 encoded image data."""

    detail: Optional[Literal["auto", "low", "high"]] = None
    """Specifies the detail level of the image.

    Learn more in the
    [Vision guide](https://platform.openai.com/docs/guides/vision#low-or-high-fidelity-image-understanding).
    """


class ChatCompletionContentPartImage(BaseModel):
    image_url: ImageURL

    type: Literal["image_url"]
    """The type of the content part."""
