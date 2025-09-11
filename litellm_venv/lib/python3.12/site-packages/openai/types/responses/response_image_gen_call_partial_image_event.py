# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseImageGenCallPartialImageEvent"]


class ResponseImageGenCallPartialImageEvent(BaseModel):
    item_id: str
    """The unique identifier of the image generation item being processed."""

    output_index: int
    """The index of the output item in the response's output array."""

    partial_image_b64: str
    """Base64-encoded partial image data, suitable for rendering as an image."""

    partial_image_index: int
    """
    0-based index for the partial image (backend is 1-based, but this is 0-based for
    the user).
    """

    sequence_number: int
    """The sequence number of the image generation item being processed."""

    type: Literal["response.image_generation_call.partial_image"]
    """The type of the event. Always 'response.image_generation_call.partial_image'."""
