import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from types import MappingProxyType
from typing import Final, cast  # noqa: TID251  # validating the openai tool union strips vendor keys from raw tools

from pydantic import ValidationError

from litellm._logging import verbose_logger
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.openai import ALL_RESPONSES_API_TOOL_PARAMS, ResponseInputParam

ADDITIONAL_TOOLS_INPUT_ITEM_TYPE: Final = "additional_tools"


class _InputItemType(LiteLLMBaseModel):
    type: str = ""


class _ToolList(LiteLLMBaseModel):
    tools: tuple[Mapping[str, object], ...] = ()


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
    if newest.get("type") != "namespace" or len(copies) == 1:
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
