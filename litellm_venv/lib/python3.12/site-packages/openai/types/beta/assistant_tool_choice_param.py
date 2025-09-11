# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

from .assistant_tool_choice_function_param import AssistantToolChoiceFunctionParam

__all__ = ["AssistantToolChoiceParam"]


class AssistantToolChoiceParam(TypedDict, total=False):
    type: Required[Literal["function", "code_interpreter", "file_search"]]
    """The type of the tool. If type is `function`, the function name must be set"""

    function: AssistantToolChoiceFunctionParam
