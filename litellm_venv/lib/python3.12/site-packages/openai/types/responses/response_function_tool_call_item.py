# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from .response_function_tool_call import ResponseFunctionToolCall

__all__ = ["ResponseFunctionToolCallItem"]


class ResponseFunctionToolCallItem(ResponseFunctionToolCall):
    id: str  # type: ignore
    """The unique ID of the function tool call."""
