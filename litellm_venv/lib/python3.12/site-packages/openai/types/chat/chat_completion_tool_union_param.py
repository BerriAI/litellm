# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union
from typing_extensions import TypeAlias

from .chat_completion_custom_tool_param import ChatCompletionCustomToolParam
from .chat_completion_function_tool_param import ChatCompletionFunctionToolParam

__all__ = ["ChatCompletionToolUnionParam"]

ChatCompletionToolUnionParam: TypeAlias = Union[
    ChatCompletionFunctionToolParam, ChatCompletionCustomToolParam
]
