# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import Annotated, TypeAlias

from ....._utils import PropertyInfo
from .function_tool_call_delta import FunctionToolCallDelta
from .file_search_tool_call_delta import FileSearchToolCallDelta
from .code_interpreter_tool_call_delta import CodeInterpreterToolCallDelta

__all__ = ["ToolCallDelta"]

ToolCallDelta: TypeAlias = Annotated[
    Union[CodeInterpreterToolCallDelta, FileSearchToolCallDelta, FunctionToolCallDelta],
    PropertyInfo(discriminator="type"),
]
