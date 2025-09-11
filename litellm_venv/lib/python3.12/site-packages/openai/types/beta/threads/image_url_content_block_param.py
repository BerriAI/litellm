# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

from .image_url_param import ImageURLParam

__all__ = ["ImageURLContentBlockParam"]


class ImageURLContentBlockParam(TypedDict, total=False):
    image_url: Required[ImageURLParam]

    type: Required[Literal["image_url"]]
    """The type of the content part."""
