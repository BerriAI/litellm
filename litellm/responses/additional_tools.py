from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast  # noqa: TID251  # validating the openai tool union strips vendor keys from raw tools

from pydantic import BaseModel, ValidationError

from litellm._logging import verbose_logger
from litellm.types.llms.openai import ALL_RESPONSES_API_TOOL_PARAMS, ResponseInputParam

ADDITIONAL_TOOLS_INPUT_ITEM_TYPE: Final = "additional_tools"
TOOL_SEARCH_OUTPUT_INPUT_ITEM_TYPE: Final = "tool_search_output"


class _InputItemType(BaseModel):
    type: str = ""


class _AdditionalToolsItem(BaseModel):
    tools: tuple[dict[str, object], ...] = ()


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


def _tools_of_item(item: object) -> tuple[ALL_RESPONSES_API_TOOL_PARAMS, ...]:
    try:
        parsed: Final = _AdditionalToolsItem.model_validate(item)
    except ValidationError:
        return ()
    return tuple(cast("ALL_RESPONSES_API_TOOL_PARAMS", tool) for tool in parsed.tools)


def _input_items_to_hoist(item: object) -> bool:
    return _is_additional_tools_item(item) or _is_tool_search_output_item(item)


def _is_tool_search_output_item(item: object) -> bool:
    try:
        return _InputItemType.model_validate(item).type == TOOL_SEARCH_OUTPUT_INPUT_ITEM_TYPE
    except ValidationError:
        return False


def _tool_key(tool: Mapping[str, object]) -> tuple[object, object]:
    return (tool.get("type"), tool.get("name"))


def hoist_additional_tools(
    input: str | ResponseInputParam,
    tools: Sequence[ALL_RESPONSES_API_TOOL_PARAMS] | None,
) -> HoistedAdditionalTools:
    existing: Final = tuple(tools or ())
    if isinstance(input, str):
        return HoistedAdditionalTools(input=input, tools=existing, hoisted=())
    items: Final = tuple(item for item in input if _input_items_to_hoist(item))
    if not items:
        return HoistedAdditionalTools(input=input, tools=existing, hoisted=())
    hoisted: list[ALL_RESPONSES_API_TOOL_PARAMS] = []  # mutable-ok: accumulator for hoisted request tools
    seen_tool_keys: set[tuple[object, object]] = {  # mutable-ok: accumulator for tool identities
        _tool_key(tool) for tool in existing
    }
    for item in items:
        for tool in _tools_of_item(item):
            if _tool_key(tool) in seen_tool_keys:
                continue
            seen_tool_keys.add(_tool_key(tool))
            hoisted.append(tool)
    hoisted_tools: Final = tuple(hoisted)
    verbose_logger.debug(
        "Responses API: hoisting %d tool(s) out of %d input item(s) into the top-level tools param.",
        len(hoisted_tools),
        len(items),
    )
    remaining_input: Final = [  # mutable-ok: JSON request payload
        item for item in input if not _is_additional_tools_item(item)
    ]
    return HoistedAdditionalTools(input=remaining_input, tools=(*existing, *hoisted_tools), hoisted=hoisted_tools)
