# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import Annotated, TypeAlias

from .._utils import PropertyInfo
from .image_gen_completed_event import ImageGenCompletedEvent
from .image_gen_partial_image_event import ImageGenPartialImageEvent

__all__ = ["ImageGenStreamEvent"]

ImageGenStreamEvent: TypeAlias = Annotated[
    Union[ImageGenPartialImageEvent, ImageGenCompletedEvent],
    PropertyInfo(discriminator="type"),
]
