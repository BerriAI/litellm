from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal, TypeVar, cast  # noqa: TID251  # validating a message union drops its untyped keys

from openai.types.responses.response_function_tool_call_param import ResponseFunctionToolCallParam
from openai.types.responses.response_input_param import ResponseInputItemParam
from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import (
    anthropic_tool_search_result,
    is_anthropic_tool_search_use,
    message_field,
    parts_of,
)
from litellm.types.llms.anthropic import (
    ANTHROPIC_TOOL_SEARCH_RESULT_CONTENT_TYPE,
    AnthropicToolReferenceBlock,
    AnthropicToolSearchResultContent,
    AnthropicToolSearchToolResultBlock,
)
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.openai import ResponseInputParam
from litellm.types.responses.main import OutputToolSearchCall, OutputToolSearchOutput

from .custom_tools import openai_shaped_tool_call_item_id, serialize_tool_call_arguments
from .tool_search import ANTHROPIC_TOOL_SEARCH_TOOL_NAME

_ItemT: Final = TypeVar("_ItemT")
_MessageT: Final = TypeVar("_MessageT")

_MAPPING_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_TOOL_SEARCH_ITEM_TYPES: Final = frozenset({"tool_search_call", "tool_search_output"})


class _ProviderToolResults(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    tool_results: tuple[object, ...] = ()


def _provider_tool_results(provider_fields: object) -> tuple[object, ...]:
    try:
        return _ProviderToolResults.model_validate(provider_fields).tool_results
    except ValidationError:
        return ()


def tool_search_results_by_call_id(
    provider_fields: Iterable[object],
) -> Mapping[str, AnthropicToolSearchToolResultBlock]:
    results: Final = map(
        anthropic_tool_search_result, chain.from_iterable(map(_provider_tool_results, provider_fields))
    )
    return MappingProxyType({result.tool_use_id: result for result in results if result is not None})


class _ToolFields(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    type: str = ""
    name: str = ""
    tools: tuple[object, ...] | None = None


@dataclass(frozen=True, slots=True)
class _LoadableTool:
    chat_name: str
    position: int
    tool: Mapping[str, object]
    member: Mapping[str, object] | None


def _tool_entry(tool: object) -> tuple[Mapping[str, object], _ToolFields] | None:
    try:
        mapping: Final = _MAPPING_ADAPTER.validate_python(tool)
        return mapping, _ToolFields.model_validate(mapping)
    except ValidationError:
        return None


def _member_chat_name(namespace: str, member: _ToolFields) -> str | None:
    match member.type:
        case "function":
            return f"{namespace}__{member.name}"
        case "custom":
            return member.name
        case _:
            return None


def _namespace_members(
    position: int, namespace: Mapping[str, object], fields: _ToolFields
) -> tuple[_LoadableTool, ...]:
    entries: Final = tuple(entry for raw in fields.tools or () if (entry := _tool_entry(raw)) is not None)
    return tuple(
        _LoadableTool(chat_name, position, namespace, member)
        for member, member_fields in entries
        if (chat_name := _member_chat_name(fields.name, member_fields)) is not None
    )


def _loadable_tools(position: int, tool: object) -> tuple[_LoadableTool, ...]:
    entry: Final = _tool_entry(tool)
    if entry is None:
        return ()
    mapping, fields = entry
    if fields.type == "namespace" and fields.tools is not None:
        return _namespace_members(position, mapping, fields)
    return (_LoadableTool(fields.name, position, mapping, None),) if fields.name else ()


def _loaded_tool(hits: Sequence[_LoadableTool]) -> Mapping[str, object]:
    members: Final = [hit.member for hit in hits if hit.member is not None]
    return {**hits[0].tool, "tools": members} if members else hits[0].tool


def loaded_tools(tool_names: Iterable[str], request_tools: Sequence[object] | None) -> tuple[Mapping[str, object], ...]:
    catalog: Final = MappingProxyType(
        {
            loadable.chat_name: loadable
            for loadable in chain.from_iterable(
                _loadable_tools(position, tool) for position, tool in enumerate(request_tools or ())
            )
        }
    )
    hits: Final = tuple(catalog[name] for name in dict.fromkeys(tool_names) if name in catalog)
    positions: Final = dict.fromkeys(hit.position for hit in hits)
    return tuple(_loaded_tool(tuple(hit for hit in hits if hit.position == position)) for position in positions)


def build_tool_search_output(
    result: AnthropicToolSearchToolResultBlock, request_tools: Sequence[object] | None
) -> OutputToolSearchOutput:
    return OutputToolSearchOutput(
        id=openai_shaped_tool_call_item_id("tool_search_output", result.tool_use_id),
        call_id=result.tool_use_id,
        execution="server",
        status="completed" if result.found_tools else "incomplete",
        tools=loaded_tools(result.tool_names, request_tools) if result.found_tools else (),
        type="tool_search_output",
    )


def _with_tool_search_output(
    item: _ItemT,
    results: Mapping[str, AnthropicToolSearchToolResultBlock],
    request_tools: Sequence[object] | None,
) -> tuple[_ItemT | OutputToolSearchOutput, ...]:
    if not isinstance(item, OutputToolSearchCall) or item.call_id is None:
        return (item,)
    result: Final = results.get(item.call_id)
    return (item,) if result is None else (item, build_tool_search_output(result, request_tools))


def with_tool_search_outputs(
    items: Sequence[_ItemT],
    results: Mapping[str, AnthropicToolSearchToolResultBlock],
    request_tools: Sequence[object] | None,
) -> tuple[_ItemT | OutputToolSearchOutput, ...]:
    return tuple(chain.from_iterable(_with_tool_search_output(item, results, request_tools) for item in items))


class _ToolSearchCallItem(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    type: Literal["tool_search_call"]
    call_id: str
    arguments: object = None


class _ToolSearchOutputItem(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    type: Literal["tool_search_output"]
    call_id: str
    status: Literal["in_progress", "completed", "incomplete"] | None = None
    tools: tuple[object, ...] = ()


def _tool_search_call(item: object) -> _ToolSearchCallItem | None:
    if message_field(item, "type") != "tool_search_call":
        return None
    try:
        return _ToolSearchCallItem.model_validate(item)
    except ValidationError:
        return None


def _tool_search_output(item: object) -> _ToolSearchOutputItem | None:
    if message_field(item, "type") != "tool_search_output":
        return None
    try:
        return _ToolSearchOutputItem.model_validate(item)
    except ValidationError:
        return None


def holds_tool_search_items(input: str | ResponseInputParam) -> bool:
    return not isinstance(input, str) and any(message_field(item, "type") in _TOOL_SEARCH_ITEM_TYPES for item in input)


def _anthropic_tool_search_result(output: _ToolSearchOutputItem) -> Mapping[str, object]:
    loaded: Final = chain.from_iterable(_loadable_tools(0, tool) for tool in output.tools)
    result: Final = AnthropicToolSearchToolResultBlock(
        tool_use_id=output.call_id,
        content=AnthropicToolSearchResultContent(
            type=ANTHROPIC_TOOL_SEARCH_RESULT_CONTENT_TYPE,
            tool_references=tuple(AnthropicToolReferenceBlock(tool_name=tool.chat_name) for tool in loaded),
        ),
    )
    return result.model_dump(mode="json")


def _replayed_item(item: ResponseInputItemParam, replayed_ids: frozenset[str]) -> ResponseInputItemParam | None:
    call: Final = _tool_search_call(item)
    if call is not None and call.call_id in replayed_ids:
        return ResponseFunctionToolCallParam(
            type="function_call",
            call_id=call.call_id,
            name=ANTHROPIC_TOOL_SEARCH_TOOL_NAME,
            arguments=serialize_tool_call_arguments(call.arguments, "{}"),
        )
    output: Final = _tool_search_output(item)
    return None if output is not None and output.call_id in replayed_ids else item


def _with_tool_search_results(message: _MessageT, results: Mapping[str, Mapping[str, object]]) -> _MessageT:
    tool_call_ids: Final = (
        message_field(tool_call, "id") for tool_call in parts_of(message_field(message, "tool_calls"))
    )
    found: Final = tuple(
        results[tool_call_id]
        for tool_call_id in tool_call_ids
        if isinstance(tool_call_id, str) and tool_call_id in results
    )
    if not found:
        return message
    fields: Final = _MAPPING_ADAPTER.validate_python(message)
    provider_fields: Final = _MAPPING_ADAPTER.validate_python(fields.get("provider_specific_fields") or {})
    tool_results: Final = (*parts_of(provider_fields.get("tool_results")), *found)
    with_results: Final = {
        **fields,
        "provider_specific_fields": {**provider_fields, "tool_results": list(tool_results)},
    }
    return cast(_MessageT, with_results)  # cast-ok: a copy of the same message dict with provider fields added


def converted_with_tool_searches_replayed(
    input: str | ResponseInputParam,
    convert: Callable[[str | ResponseInputParam], Sequence[_MessageT]],
) -> Sequence[_MessageT]:
    if isinstance(input, str) or not holds_tool_search_items(input):
        return convert(input)
    outputs: Final = MappingProxyType(
        {
            output.call_id: output
            for output in map(_tool_search_output, input)
            if output is not None and output.status in (None, "completed")
        }
    )
    replayed_ids: Final = frozenset(
        call.call_id
        for call in map(_tool_search_call, input)
        if call is not None
        and call.call_id in outputs
        and is_anthropic_tool_search_use(call.call_id, ANTHROPIC_TOOL_SEARCH_TOOL_NAME)
    )
    if not replayed_ids:
        return convert(input)
    replayed_input: Final = [kept for item in input if (kept := _replayed_item(item, replayed_ids)) is not None]
    results: Final = MappingProxyType(
        {call_id: _anthropic_tool_search_result(outputs[call_id]) for call_id in replayed_ids}
    )
    return tuple(_with_tool_search_results(message, results) for message in convert(replayed_input))
