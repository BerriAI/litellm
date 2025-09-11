# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union
from typing_extensions import Literal, TypeAlias

from ..shared_params.response_format_text import ResponseFormatText
from ..shared_params.response_format_json_object import ResponseFormatJSONObject
from ..shared_params.response_format_json_schema import ResponseFormatJSONSchema

__all__ = ["AssistantResponseFormatOptionParam"]

AssistantResponseFormatOptionParam: TypeAlias = Union[
    Literal["auto"],
    ResponseFormatText,
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
]
