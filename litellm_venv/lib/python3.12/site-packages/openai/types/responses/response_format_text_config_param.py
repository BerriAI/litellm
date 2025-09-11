# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union
from typing_extensions import TypeAlias

from ..shared_params.response_format_text import ResponseFormatText
from ..shared_params.response_format_json_object import ResponseFormatJSONObject
from .response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)

__all__ = ["ResponseFormatTextConfigParam"]

ResponseFormatTextConfigParam: TypeAlias = Union[
    ResponseFormatText,
    ResponseFormatTextJSONSchemaConfigParam,
    ResponseFormatJSONObject,
]
