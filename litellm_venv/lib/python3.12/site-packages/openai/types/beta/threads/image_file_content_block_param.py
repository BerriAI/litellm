# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

from .image_file_param import ImageFileParam

__all__ = ["ImageFileContentBlockParam"]


class ImageFileContentBlockParam(TypedDict, total=False):
    image_file: Required[ImageFileParam]

    type: Required[Literal["image_file"]]
    """Always `image_file`."""
