# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

__all__ = ["ModerationImageURLInputParam", "ImageURL"]


class ImageURL(TypedDict, total=False):
    url: Required[str]
    """Either a URL of the image or the base64 encoded image data."""


class ModerationImageURLInputParam(TypedDict, total=False):
    image_url: Required[ImageURL]
    """Contains either an image URL or a data URL for a base64 encoded image."""

    type: Required[Literal["image_url"]]
    """Always `image_url`."""
