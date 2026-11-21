import asyncio
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NoReturn

from fastapi import HTTPException
from pydantic import TypeAdapter
from typing_extensions import assert_never

from litellm.proxy.mcp_server.utils import (
    MAX_STRUCTURED_CONTENT_SCAN_DEPTH,
    JSONLeafPath,
    json_string_leaves,
    with_json_string_leaves,
)
from litellm.responses.additional_tools import ADDITIONAL_TOOLS_INPUT_ITEM_TYPE
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_MAX_CONCURRENT_TOOL_SCANS: Final = 8
_JSON_ARRAY: Final = TypeAdapter(tuple[object, ...])
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_TOOL_DEFINITION_FIELDS: Final = ("tools", "functions")
_INPUT_ITEM_TYPES_WITH_TOOLS: Final = frozenset({ADDITIONAL_TOOLS_INPUT_ITEM_TYPE, "tool_search_output"})


@dataclass(frozen=True, slots=True)
class _ToolNotScanned:
    reason: str


@dataclass(frozen=True, slots=True)
class _RequestTool:
    field: str
    path: JSONLeafPath
    definition: object


def _raise_public(failure: _ToolNotScanned) -> NoReturn:
    match failure:
        case _ToolNotScanned(reason=reason):
            raise HTTPException(status_code=400, detail={"error": f"Content blocked: {reason}"})
        case _:
            assert_never(failure)


def _json_array(value: object) -> tuple[object, ...]:
    return _JSON_ARRAY.validate_python(value) if isinstance(value, list) else ()


def _top_level_tools(request_data: Mapping[str, object], field: str) -> Iterator[_RequestTool]:
    return (
        _RequestTool(field, (index,), definition)
        for index, definition in enumerate(_json_array(request_data.get(field)))
    )


def _input_item_tools(index: int, item: object) -> Iterator[_RequestTool]:
    fields: Final = _JSON_OBJECT.validate_python(item) if isinstance(item, dict) else {}
    if fields.get("type") not in _INPUT_ITEM_TYPES_WITH_TOOLS:
        return iter(())
    return (
        _RequestTool("input", (index, "tools", position), definition)
        for position, definition in enumerate(_json_array(fields.get("tools")))
    )


def _request_tools(request_data: Mapping[str, object]) -> tuple[_RequestTool, ...]:
    input_items: Final = enumerate(_json_array(request_data.get("input")))
    return (
        *chain.from_iterable(_top_level_tools(request_data, field) for field in _TOOL_DEFINITION_FIELDS),
        *chain.from_iterable(_input_item_tools(index, item) for index, item in input_items),
    )


def _description_leaves(tool: _RequestTool) -> tuple[tuple[JSONLeafPath, str], ...] | _ToolNotScanned:
    leaves: Final = json_string_leaves(tool.definition, tool.path)
    if leaves is None:
        return _ToolNotScanned(
            f"a tool definition exceeds the maximum nesting depth of {MAX_STRUCTURED_CONTENT_SCAN_DEPTH} "
            "and cannot be scanned by the configured guardrail"
        )
    return tuple((path, text) for path, text in leaves if path[-1:] == ("description",) and text.strip())


def _masked_texts(guarded: GenericGuardrailAPIInputs, scanned: int) -> Sequence[str] | _ToolNotScanned | None:
    masked: Final = guarded.get("texts")
    if masked is not None and len(masked) != scanned:
        return _ToolNotScanned("the guardrail returned a different number of tool descriptions than it was given")
    return masked


async def scanned_request_tools(
    guardrail: "CustomGuardrail",
    request_data: dict[str, object],  # mutable-ok: guardrails annotate the hook's request dict in place
    logging_obj: "LiteLLMLoggingObj | None",
) -> Mapping[str, object]:
    if guardrail.scan_tool_descriptions_as_text is not True:
        return MappingProxyType({})
    limiter: Final = asyncio.Semaphore(_MAX_CONCURRENT_TOOL_SCANS)

    async def masked_leaves(tool: _RequestTool) -> tuple[str, Mapping[JSONLeafPath, str]] | _ToolNotScanned:
        leaves: Final = _description_leaves(tool)
        if isinstance(leaves, _ToolNotScanned):
            return leaves
        if not leaves:
            return tool.field, MappingProxyType({})
        async with limiter:
            guarded: Final = await guardrail.apply_guardrail(  # pyright: ignore[reportUnknownMemberType]  # apply_guardrail's request_data parameter is an untyped dict
                inputs=GenericGuardrailAPIInputs(texts=[text for _, text in leaves]),
                request_data=request_data,
                input_type="request",
                logging_obj=logging_obj,
            )
        masked: Final = _masked_texts(guarded, len(leaves))
        if isinstance(masked, _ToolNotScanned):
            return masked
        if masked is None:
            return tool.field, MappingProxyType({})
        return tool.field, MappingProxyType({path: new for (path, old), new in zip(leaves, masked) if new != old})

    results: Final = await asyncio.gather(*(masked_leaves(tool) for tool in _request_tools(request_data)))
    failure: Final = next((result for result in results if isinstance(result, _ToolNotScanned)), None)
    if failure is not None:
        _raise_public(failure)
    changed: Final = tuple(result for result in results if isinstance(result, tuple) and result[1])
    return MappingProxyType(
        {
            field: with_json_string_leaves(request_data[field], _field_replacements(field, changed))
            for field in frozenset(name for name, _ in changed)
        }
    )


def _field_replacements(
    field: str, changed: Sequence[tuple[str, Mapping[JSONLeafPath, str]]]
) -> Mapping[JSONLeafPath, str]:
    return MappingProxyType(dict(chain.from_iterable(leaves.items() for name, leaves in changed if name == field)))
