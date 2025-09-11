# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ....._models import BaseModel

__all__ = ["CodeInterpreterOutputImage", "Image"]


class Image(BaseModel):
    file_id: Optional[str] = None
    """
    The [file](https://platform.openai.com/docs/api-reference/files) ID of the
    image.
    """


class CodeInterpreterOutputImage(BaseModel):
    index: int
    """The index of the output in the outputs array."""

    type: Literal["image"]
    """Always `image`."""

    image: Optional[Image] = None
