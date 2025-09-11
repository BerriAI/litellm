# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseImageGenCallInProgressEvent"]


class ResponseImageGenCallInProgressEvent(BaseModel):
    item_id: str
    """The unique identifier of the image generation item being processed."""

    output_index: int
    """The index of the output item in the response's output array."""

    sequence_number: int
    """The sequence number of the image generation item being processed."""

    type: Literal["response.image_generation_call.in_progress"]
    """The type of the event. Always 'response.image_generation_call.in_progress'."""
