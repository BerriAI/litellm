# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List
from typing_extensions import Literal

from .tool_call import ToolCall
from ....._models import BaseModel

__all__ = ["ToolCallsStepDetails"]


class ToolCallsStepDetails(BaseModel):
    tool_calls: List[ToolCall]
    """An array of tool calls the run step was involved in.

    These can be associated with one of three types of tools: `code_interpreter`,
    `file_search`, or `function`.
    """

    type: Literal["tool_calls"]
    """Always `tool_calls`."""
