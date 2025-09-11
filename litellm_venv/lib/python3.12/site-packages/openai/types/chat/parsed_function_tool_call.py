# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional

from .chat_completion_message_function_tool_call import (
    Function,
    ChatCompletionMessageFunctionToolCall,
)

__all__ = ["ParsedFunctionToolCall", "ParsedFunction"]

# we need to disable this check because we're overriding properties
# with subclasses of their types which is technically unsound as
# properties can be mutated.
# pyright: reportIncompatibleVariableOverride=false


class ParsedFunction(Function):
    parsed_arguments: Optional[object] = None
    """
    The arguments to call the function with.

    If you used `openai.pydantic_function_tool()` then this will be an
    instance of the given `BaseModel`.

    Otherwise, this will be the parsed JSON arguments.
    """


class ParsedFunctionToolCall(ChatCompletionMessageFunctionToolCall):
    function: ParsedFunction
    """The function that the model called."""
