"""Opt-in pre_call scan of the description strings in a request's tool definitions.

Every API surface (Chat Completions, Responses, Anthropic Messages) carries its tool
definitions in the request's top-level ``tools`` list, plus Chat Completions' legacy
``functions`` list, so walking each definition for string
leaves keyed ``description`` covers tool and parameter descriptions without per-surface
parsing. Each tool is one ``apply_guardrail`` call, mirroring the MCP ``tools/list`` scan.
"""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NoReturn

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.proxy.mcp_server.utils import (
    MAX_STRUCTURED_CONTENT_SCAN_DEPTH,
    JSONLeafPath,
    json_string_leaves,
    with_json_string_leaves,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_MAX_CONCURRENT_TOOL_SCANS: Final = 8
_TOOLS: Final = TypeAdapter(tuple[object, ...])
_TOOL_DEFINITION_FIELDS: Final = ("tools", "functions")


@dataclass(frozen=True, slots=True)
class _ToolNotScanned:
    reason: str


def _raise_public(failure: _ToolNotScanned) -> NoReturn:
    match failure:
        case _ToolNotScanned(reason=reason):
            raise HTTPException(status_code=400, detail={"error": f"Content blocked: {reason}"})
        case _:
            assert_never(failure)


def _description_leaves(tool: object) -> tuple[tuple[JSONLeafPath, str], ...] | _ToolNotScanned:
    leaves: Final = json_string_leaves(tool)
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


def _request_tools(request_data: Mapping[str, object], field: str) -> tuple[object, ...]:
    try:
        return _TOOLS.validate_python(request_data.get(field) or ())
    except ValidationError:
        return ()


async def scanned_request_tools(
    guardrail: "CustomGuardrail",
    request_data: dict[str, object],  # mutable-ok: guardrails annotate the hook's request dict in place
    logging_obj: "LiteLLMLoggingObj | None",
) -> Mapping[str, tuple[object, ...]]:
    """Each tool definition field (``tools``, legacy ``functions``) whose descriptions were masked.

    A guardrail blocks by raising, which surfaces as the usual pre_call HTTP 400; a tool that
    cannot be scanned is rejected the same way.
    """
    if guardrail.scan_tool_descriptions_as_text is not True:
        return MappingProxyType({})
    limiter: Final = asyncio.Semaphore(_MAX_CONCURRENT_TOOL_SCANS)

    async def scanned(tool: object) -> object | _ToolNotScanned:
        leaves: Final = _description_leaves(tool)
        if isinstance(leaves, _ToolNotScanned):
            return leaves
        if not leaves:
            return tool
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
            return tool
        replacements: Final = {path: new for (path, old), new in zip(leaves, masked) if new != old}
        return with_json_string_leaves(tool, replacements) if replacements else tool

    async def scanned_field(field: str) -> tuple[str, tuple[object, ...]] | _ToolNotScanned | None:
        tools: Final = _request_tools(request_data, field)
        results: Final = tuple(await asyncio.gather(*(scanned(tool) for tool in tools)))
        failure: Final = next((result for result in results if isinstance(result, _ToolNotScanned)), None)
        if failure is not None:
            return failure
        return None if all(new is old for new, old in zip(results, tools)) else (field, results)

    fields: Final = await asyncio.gather(*(scanned_field(field) for field in _TOOL_DEFINITION_FIELDS))
    failure: Final = next((field for field in fields if isinstance(field, _ToolNotScanned)), None)
    if failure is not None:
        _raise_public(failure)
    return MappingProxyType(dict(field for field in fields if isinstance(field, tuple)))
