# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union
from typing_extensions import TypeAlias

from .moderation_text_input_param import ModerationTextInputParam
from .moderation_image_url_input_param import ModerationImageURLInputParam

__all__ = ["ModerationMultiModalInputParam"]

ModerationMultiModalInputParam: TypeAlias = Union[
    ModerationImageURLInputParam, ModerationTextInputParam
]
