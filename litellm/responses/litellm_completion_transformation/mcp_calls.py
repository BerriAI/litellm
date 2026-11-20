from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal

from openai.types.responses.response_output_item import McpCall
from pydantic import ConfigDict, ValidationError

from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import message_field, parts_of
from litellm.types.llms.anthropic import ANTHROPIC_MCP_TOOL_USE_ID_PREFIX
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.openai import (
    ChatCompletionAssistantToolCall,
    ChatCompletionResponseMessage,
    ChatCompletionToolCallFunctionChunk,
)

from .custom_tools import serialize_tool_call_arguments

_MCP_CALL_ITEM_ID_PREFIX: Final = "mcp_"
_DEFERRED_RESULT_ITEM_ID_SUFFIX: Final = "_result"


class _McpServer(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    server_name: str


class _McpToolUseFunction(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    arguments: object = None
    provider_specific_fields: _McpServer


class McpToolUse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    id: str
    function: _McpToolUseFunction


class _ResultBlock(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    text: str = ""


class McpToolResult(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    type: Literal["mcp_tool_result"]
    tool_use_id: str
    is_error: bool = False
    content: str | tuple[_ResultBlock, ...] = ()


class _McpCallInputItem(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["mcp_call"]
    id: str
    name: str
    arguments: object = None
    server_label: str
    output: str | None = None
    error: str | None = None


def is_anthropic_mcp_call_id(call_id: str) -> bool:
    return call_id.startswith(ANTHROPIC_MCP_TOOL_USE_ID_PREFIX)


def anthropic_mcp_call_id(item_id: str) -> str:
    return item_id.removeprefix(_MCP_CALL_ITEM_ID_PREFIX).removesuffix(_DEFERRED_RESULT_ITEM_ID_SUFFIX)


def _mcp_tool_use(call: object) -> McpToolUse | None:
    if not is_anthropic_mcp_call_id(str(message_field(call, "id") or "")):
        return None
    try:
        return McpToolUse.model_validate(call)
    except ValidationError:
        return None


def mcp_tool_uses(holder: object) -> tuple[McpToolUse, ...]:
    server_tool_calls: Final = message_field(message_field(holder, "provider_specific_fields"), "server_tool_calls")
    return tuple(call for raw in parts_of(server_tool_calls) if (call := _mcp_tool_use(raw)) is not None)


def _mcp_tool_result(block: object) -> McpToolResult | None:
    if message_field(block, "type") != "mcp_tool_result":
        return None
    try:
        return McpToolResult.model_validate(block)
    except ValidationError:
        return None


def mcp_tool_results_by_call_id(provider_fields: Iterable[object]) -> Mapping[str, McpToolResult]:
    blocks: Final = chain.from_iterable(parts_of(message_field(fields, "tool_results")) for fields in provider_fields)
    return MappingProxyType(
        {result.tool_use_id: result for block in blocks if (result := _mcp_tool_result(block)) is not None}
    )


def _result_text(result: McpToolResult) -> str:
    if isinstance(result.content, str):
        return result.content
    return "\n".join(block.text for block in result.content if block.type == "text")


def build_mcp_call(call: McpToolUse, result: McpToolResult | None) -> McpCall:
    pending: Final = McpCall(
        id=f"{_MCP_CALL_ITEM_ID_PREFIX}{call.id}",
        type="mcp_call",
        name=call.function.name,
        arguments=serialize_tool_call_arguments(call.function.arguments),
        server_label=call.function.provider_specific_fields.server_name,
        status="in_progress",
    )
    if result is None:
        return pending
    if result.is_error:
        return pending.model_copy(update={"status": "failed", "error": _result_text(result)})
    return pending.model_copy(update={"status": "completed", "output": _result_text(result)})


def build_deferred_mcp_call(call: McpToolUse, result: McpToolResult | None) -> McpCall:
    item: Final = build_mcp_call(call, result)
    return item.model_copy(update={"id": f"{item.id}{_DEFERRED_RESULT_ITEM_ID_SUFFIX}"})


def _results_of(messages: Sequence[object]) -> Mapping[str, McpToolResult]:
    return mcp_tool_results_by_call_id(message_field(message, "provider_specific_fields") for message in messages)


def requested_mcp_tool_uses(messages: object) -> Mapping[str, McpToolUse]:
    calls: Final = chain.from_iterable(map(mcp_tool_uses, parts_of(messages)))
    return MappingProxyType({call.id: call for call in calls})


def mcp_calls_by_call_id(messages: Sequence[object]) -> Mapping[str, McpCall]:
    results: Final = _results_of(messages)
    calls: Final = chain.from_iterable(map(mcp_tool_uses, messages))
    return MappingProxyType({call.id: build_mcp_call(call, results.get(call.id)) for call in calls})


def deferred_mcp_calls(messages: Sequence[object], requested: Mapping[str, McpToolUse]) -> tuple[McpCall, ...]:
    own_call_ids: Final = frozenset(call.id for call in chain.from_iterable(map(mcp_tool_uses, messages)))
    return tuple(
        build_deferred_mcp_call(requested[call_id], result)
        for call_id, result in _results_of(messages).items()
        if call_id not in own_call_ids and call_id in requested
    )


def holds_mcp_call_items(request_input: str | Iterable[object]) -> bool:
    return not isinstance(request_input, str) and any(
        message_field(item, "type") == "mcp_call" for item in request_input
    )


def _mcp_call_input_item(input_item: Mapping[str, object]) -> _McpCallInputItem | None:
    if input_item.get("type") != "mcp_call":
        return None
    try:
        return _McpCallInputItem.model_validate(input_item)
    except ValidationError:
        return None


def _replayed_result_content(text: str) -> tuple[Mapping[str, str], ...]:
    return ({"type": "text", "text": text},) if text else ()


def _replayed_mcp_tool_result(call_id: str, item: _McpCallInputItem) -> Mapping[str, object] | None:
    if item.error is not None:
        return {
            "type": "mcp_tool_result",
            "tool_use_id": call_id,
            "is_error": True,
            "content": list(_replayed_result_content(item.error)),
        }
    if item.output is not None:
        return {
            "type": "mcp_tool_result",
            "tool_use_id": call_id,
            "is_error": False,
            "content": list(_replayed_result_content(item.output)),
        }
    return None


def replayed_mcp_call_message(input_item: Mapping[str, object]) -> ChatCompletionResponseMessage | None:
    item: Final = _mcp_call_input_item(input_item)
    if item is None:
        return None
    call_id: Final = anthropic_mcp_call_id(item.id)
    if not is_anthropic_mcp_call_id(call_id):
        return None
    server_tool_call: Final = ChatCompletionAssistantToolCall(
        id=call_id,
        type="function",
        function=ChatCompletionToolCallFunctionChunk(
            name=item.name,
            arguments=serialize_tool_call_arguments(item.arguments, "{}"),
            provider_specific_fields={"server_name": item.server_label},
        ),
    )
    result: Final = _replayed_mcp_tool_result(call_id, item)
    return ChatCompletionResponseMessage(
        role="assistant",
        content=None,
        provider_specific_fields={
            "server_tool_calls": [server_tool_call],
            **({} if result is None else {"tool_results": [result]}),
        },
    )
