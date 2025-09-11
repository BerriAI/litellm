# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import Annotated, TypeAlias

from ...._utils import PropertyInfo
from .text_delta_block import TextDeltaBlock
from .refusal_delta_block import RefusalDeltaBlock
from .image_url_delta_block import ImageURLDeltaBlock
from .image_file_delta_block import ImageFileDeltaBlock

__all__ = ["MessageContentDelta"]

MessageContentDelta: TypeAlias = Annotated[
    Union[ImageFileDeltaBlock, TextDeltaBlock, RefusalDeltaBlock, ImageURLDeltaBlock],
    PropertyInfo(discriminator="type"),
]
