# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel
from ..shared.function_definition import FunctionDefinition

__all__ = ["ChatCompletionFunctionTool"]


class ChatCompletionFunctionTool(BaseModel):
    function: FunctionDefinition

    type: Literal["function"]
    """The type of the tool. Currently, only `function` is supported."""
