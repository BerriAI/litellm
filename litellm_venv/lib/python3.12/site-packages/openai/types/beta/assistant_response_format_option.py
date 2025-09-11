# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import Literal, TypeAlias

from ..shared.response_format_text import ResponseFormatText
from ..shared.response_format_json_object import ResponseFormatJSONObject
from ..shared.response_format_json_schema import ResponseFormatJSONSchema

__all__ = ["AssistantResponseFormatOption"]

AssistantResponseFormatOption: TypeAlias = Union[
    Literal["auto"],
    ResponseFormatText,
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
]
