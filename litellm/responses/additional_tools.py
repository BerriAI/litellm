from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast  # noqa: TID251  # validating the openai tool union strips vendor keys from raw tools

from pydantic import BaseModel, ValidationError

from litellm._logging import verbose_logger
from litellm.types.llms.openai import ALL_RESPONSES_API_TOOL_PARAMS, ResponseInputParam

ADDITIONAL_TOOLS_INPUT_ITEM_TYPE: Final = "additional_tools"


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
    return tuple(
        cast(
            "ALL_RESPONSES_API_TOOL_PARAMS", tool
        )  # cast-ok: nested tools carry the same raw tool JSON as top-level tools
        for tool in parsed.tools
    )


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
    return HoistedAdditionalTools(input=remaining_input, tools=(*existing, *hoisted), hoisted=hoisted)
