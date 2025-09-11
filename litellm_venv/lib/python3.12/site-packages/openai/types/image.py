# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional

from .._models import BaseModel

__all__ = ["Image"]


class Image(BaseModel):
    b64_json: Optional[str] = None
    """The base64-encoded JSON of the generated image.

    Default value for `gpt-image-1`, and only present if `response_format` is set to
    `b64_json` for `dall-e-2` and `dall-e-3`.
    """

    revised_prompt: Optional[str] = None
    """For `dall-e-3` only, the revised prompt that was used to generate the image."""

    url: Optional[str] = None
    """
    When using `dall-e-2` or `dall-e-3`, the URL of the generated image if
    `response_format` is set to `url` (default value). Unsupported for
    `gpt-image-1`.
    """
