# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["ImageEditPartialImageEvent"]


class ImageEditPartialImageEvent(BaseModel):
    b64_json: str
    """Base64-encoded partial image data, suitable for rendering as an image."""

    background: Literal["transparent", "opaque", "auto"]
    """The background setting for the requested edited image."""

    created_at: int
    """The Unix timestamp when the event was created."""

    output_format: Literal["png", "webp", "jpeg"]
    """The output format for the requested edited image."""

    partial_image_index: int
    """0-based index for the partial image (streaming)."""

    quality: Literal["low", "medium", "high", "auto"]
    """The quality setting for the requested edited image."""

    size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"]
    """The size of the requested edited image."""

    type: Literal["image_edit.partial_image"]
    """The type of the event. Always `image_edit.partial_image`."""
