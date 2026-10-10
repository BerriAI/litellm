from collections.abc import Mapping, Sequence
from typing import Final

from typing_extensions import assert_never

import litellm
from litellm._internal_context import is_internal_call
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.responses.tool_search.lifting import lift_tool_search_calls, lift_tool_search_stream
from litellm.responses.tool_search.lowering import (
    TOOL_SEARCH_FUNCTION_NAME,
    LoweredToolSearchRequest,
    ToolSearchFunctionNameTaken,
    lower_tool_search_request,
)
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIResponse, ToolChoice


def _lowered_call_kwargs(
    input: str | ResponseInputParam,
    model: str,
    tools: Sequence[object] | None,
    tool_choice: ToolChoice | None,
    call_kwargs: Mapping[str, object],
) -> dict[str, object]:
    custom_llm_provider: Final = call_kwargs.get("custom_llm_provider")
    lowering: Final = lower_tool_search_request(input=input, tools=tools, tool_choice=tool_choice)
    match lowering:
        case LoweredToolSearchRequest():
            return {
                **call_kwargs,
                "model": model,
                "input": lowering.input,
                "tools": list(lowering.tools),
                "tool_choice": lowering.tool_choice,
            }
        case ToolSearchFunctionNameTaken():
            raise litellm.BadRequestError(
                message=(
                    f"A function tool named '{TOOL_SEARCH_FUNCTION_NAME}' can't be declared alongside a client "
                    "tool_search tool, because this deployment receives tool_search as that function"
                ),
                model=model,
                llm_provider=custom_llm_provider if isinstance(custom_llm_provider, str) else "",
            )
    return assert_never(lowering)


def _lifted(result: object) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
    if isinstance(result, BaseResponsesAPIStreamingIterator):
        return lift_tool_search_stream(result)
    if isinstance(result, ResponsesAPIResponse):
        return lift_tool_search_calls(result)
    raise TypeError(f"Unexpected Responses API result: {type(result).__name__}")


async def aresponses_with_lowered_tool_search(
    input: str | ResponseInputParam,
    model: str,
    tools: Sequence[object] | None,
    tool_choice: ToolChoice | None,
    call_kwargs: Mapping[str, object],
) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
    from litellm.responses.dispatch import aresponses

    lowered_kwargs: Final = _lowered_call_kwargs(input, model, tools, tool_choice, call_kwargs)
    internal_call: Final = is_internal_call.set(True)
    try:
        result: Final = await aresponses(**lowered_kwargs)
    finally:
        is_internal_call.reset(internal_call)
    return _lifted(result)


def responses_with_lowered_tool_search(
    input: str | ResponseInputParam,
    model: str,
    tools: Sequence[object] | None,
    tool_choice: ToolChoice | None,
    call_kwargs: Mapping[str, object],
) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
    from litellm.responses.dispatch import responses

    lowered_kwargs: Final = _lowered_call_kwargs(input, model, tools, tool_choice, call_kwargs)
    internal_call: Final = is_internal_call.set(True)
    try:
        result: Final = responses(**lowered_kwargs)
    finally:
        is_internal_call.reset(internal_call)
    return _lifted(result)
