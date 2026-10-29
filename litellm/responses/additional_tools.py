import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, groupby
from types import MappingProxyType
from typing import (
    Annotated,
    Final,
    Literal,
    cast,  # noqa: TID251  # validating the openai tool union strips vendor keys from raw tools
)

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.openai import ALL_RESPONSES_API_TOOL_PARAMS, ResponseInputParam

ADDITIONAL_TOOLS_INPUT_ITEM_TYPE: Final = "additional_tools"
_NAMESPACE_TOOL_TYPE: Final = "namespace"
_ITEMS: Final = TypeAdapter(tuple[object, ...])
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])


class _InputItemType(LiteLLMBaseModel):
    type: str = ""


class _ToolList(LiteLLMBaseModel):
    tools: tuple[Mapping[str, object], ...] = ()


class _NamespacedFunctionCall(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["function_call"]
    name: str
    namespace: Annotated[str, Field(min_length=1)]


@dataclass(frozen=True, slots=True)
class HoistedAdditionalTools:
    input: str | ResponseInputParam
    tools: tuple[ALL_RESPONSES_API_TOOL_PARAMS, ...]
    hoisted: tuple[ALL_RESPONSES_API_TOOL_PARAMS, ...]


def _is_additional_tools_item(item: object) -> bool:
    try:
        return _InputItemType.model_validate(item).type == ADDITIONAL_TOOLS_INPUT_ITEM_TYPE
    except ValidationError:
        return False


def _tool_list_of(value: object) -> tuple[Mapping[str, object], ...]:
    try:
        return _ToolList.model_validate(value).tools
    except ValidationError:
        return ()


def _tools_of_item(item: object) -> tuple[ALL_RESPONSES_API_TOOL_PARAMS, ...]:
    return tuple(cast("ALL_RESPONSES_API_TOOL_PARAMS", tool) for tool in _tool_list_of(item))


def _tool_identity(tool: Mapping[str, object]) -> str:
    name: Final = tool.get("name")
    if isinstance(name, str) and name:
        return f"{tool.get('type')}:{name}"
    return json.dumps(tool, sort_keys=True, default=str)


def _members_of_copies(copies: Sequence[Mapping[str, object]]) -> Iterator[Mapping[str, object]]:
    for namespace in copies:
        yield from _tool_list_of(namespace)


def _merged(copies: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    newest: Final = copies[-1]
    if newest.get("type") != _NAMESPACE_TOOL_TYPE or len(copies) == 1:
        return newest
    members: Final = tuple(_members_of_copies(copies))
    return {**newest, "tools": list(_unique_tools(members))} if members else newest


def _identity_of(keyed_tool: tuple[str, Mapping[str, object]]) -> str:
    return keyed_tool[0]


def _unique_tools(tools: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    """Codex re-sends its tool list in the input on every turn, so the same tool arrives many times.
    Keep each tool once, at its first position, with the newest copy's definition, and merge the
    members of a namespace sent more than once."""
    keyed: Final = tuple((_tool_identity(tool), tool) for tool in tools)
    copies_by_identity: Final = MappingProxyType(
        {
            identity: tuple(tool for _, tool in group)
            for identity, group in groupby(sorted(keyed, key=_identity_of), key=_identity_of)
        }
    )
    first_seen_order: Final = tuple(dict.fromkeys(identity for identity, _ in keyed))
    return tuple(_merged(copies_by_identity[identity]) for identity in first_seen_order)


def hoist_additional_tools(
    input: str | ResponseInputParam,
    tools: Sequence[ALL_RESPONSES_API_TOOL_PARAMS] | None,
) -> HoistedAdditionalTools:
    existing: Final = tuple(tools or ())
    if isinstance(input, str):
        return HoistedAdditionalTools(input=input, tools=existing, hoisted=())
    items: Final = tuple(item for item in input if _is_additional_tools_item(item))
    if not items:
        return HoistedAdditionalTools(input=input, tools=existing, hoisted=())
    hoisted: Final = tuple(tool for item in items for tool in _tools_of_item(item))
    verbose_logger.debug(
        "Responses API: hoisting %d tool(s) out of %d 'additional_tools' input item(s) into the top-level tools param.",
        len(hoisted),
        len(items),
    )
    remaining_input: Final = [item for item in input if not _is_additional_tools_item(item)]
    unique: Final = tuple(
        cast("ALL_RESPONSES_API_TOOL_PARAMS", tool)  # cast-ok: a merged namespace is rebuilt from the same tool dicts
        for tool in _unique_tools((*existing, *hoisted))
    )
    return HoistedAdditionalTools(input=remaining_input, tools=unique, hoisted=hoisted)


def with_additional_tools_hoisted(request: Mapping[str, object]) -> Mapping[str, object]:
    request_input: Final = request.get("input")
    if not isinstance(request_input, list):
        return request
    input_items: Final = cast(ResponseInputParam, request_input)  # cast-ok: forwarded as sent, as inference does
    top_level_tools: Final = cast(  # cast-ok: tools are forwarded as sent, as inference does
        "tuple[ALL_RESPONSES_API_TOOL_PARAMS, ...]", _ITEMS.validate_python(request.get("tools") or ())
    )
    hoisted: Final = hoist_additional_tools(input_items, top_level_tools)
    if not hoisted.hoisted:
        return request
    return {**request, "input": hoisted.input, "tools": list(hoisted.tools)}


def _flattened_member(
    namespace: str, member: Mapping[str, object], flat_names: Mapping[tuple[str, str], str]
) -> Mapping[str, object]:
    flat_name: Final = flat_names.get((namespace, str(member.get("name") or "")))
    return member if flat_name is None else {**member, "name": flat_name}


def _members_as_tools(
    tool: Mapping[str, object], flat_names: Mapping[tuple[str, str], str]
) -> tuple[Mapping[str, object], ...]:
    if tool.get("type") != _NAMESPACE_TOOL_TYPE:
        return (tool,)
    namespace: Final = str(tool.get("name") or "")
    return tuple(_flattened_member(namespace, member, flat_names) for member in _tool_list_of(tool))


def _flattened_call(item: object, flat_names: Mapping[tuple[str, str], str]) -> object:
    try:
        call: Final = _NamespacedFunctionCall.model_validate(item)
    except ValidationError:
        return item
    flat_name: Final = flat_names.get((call.namespace, call.name))
    if flat_name is None:
        return item
    fields: Final = _JSON_OBJECT.validate_python(item)
    return {**{key: value for key, value in fields.items() if key != "namespace"}, "name": flat_name}


def with_namespaces_flattened(
    request: Mapping[str, object], namespaced_names: Mapping[str, tuple[str, str]]
) -> Mapping[str, object]:
    if not namespaced_names:
        return request
    flat_names: Final = MappingProxyType({namespaced: flat for flat, namespaced in namespaced_names.items()})
    tools: Final = chain.from_iterable(_members_as_tools(tool, flat_names) for tool in _tool_list_of(request))
    request_input: Final = request.get("input")
    replayed: Final = (
        {"input": [_flattened_call(item, flat_names) for item in _ITEMS.validate_python(request_input)]}
        if isinstance(request_input, list)
        else {}
    )
    return {**request, "tools": list(tools), **replayed}
