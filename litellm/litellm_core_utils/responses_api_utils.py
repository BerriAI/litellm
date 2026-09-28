from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Protocol, cast, runtime_checkable


@runtime_checkable
class _FunctionToolLike(Protocol):
    type: str


def has_function_tool(tools: object) -> bool:
    if not isinstance(tools, Sequence):
        return False
    typed_tools: Final = cast(Sequence[object], tools)  # cast-ok: a runtime Sequence contains Python objects
    return any(_is_function_tool(tool) for tool in typed_tools)


def _is_function_tool(tool: object) -> bool:
    if not isinstance(tool, Mapping):
        return isinstance(tool, _FunctionToolLike) and tool.type == "function"
    tool_mapping: Final = cast(Mapping[str, object], tool)  # cast-ok: OpenAI tool mappings have string keys
    function: Final = tool_mapping.get("function")
    return tool_mapping.get("type") == "function" and (isinstance(function, Mapping) or "name" in tool_mapping)


def reasoning_effort_is_active(
    reasoning_effort: object,
    supports_none_reasoning_effort: bool | None = None,
) -> bool:
    if isinstance(reasoning_effort, Mapping):
        effort_mapping: Final = cast(  # cast-ok: reasoning payload keys are strings
            Mapping[str, object], reasoning_effort
        )
        return (
            effort_mapping.get("effort") != "none"
            or effort_mapping.get("summary") is not None
            or supports_none_reasoning_effort is False
        )
    return reasoning_effort != "none" or supports_none_reasoning_effort is False


def peek_reasoning_summary_aliases(optional_params: Mapping[str, object]) -> object | None:
    if "reasoningSummary" in optional_params:
        return optional_params["reasoningSummary"]
    if "reasoning_summary" in optional_params:
        return optional_params["reasoning_summary"]
    extra_body: Final = optional_params.get("extra_body")
    if isinstance(extra_body, Mapping):
        extra_body_mapping: Final = cast(  # cast-ok: extra_body uses JSON object keys
            Mapping[str, object], extra_body
        )
        if "reasoningSummary" in extra_body_mapping:
            return extra_body_mapping["reasoningSummary"]
        if "reasoning_summary" in extra_body_mapping:
            return extra_body_mapping["reasoning_summary"]
    return None


def filter_additional_drop_params(
    request_params: Mapping[str, object],
    additional_drop_params: Sequence[str] | None,
) -> Mapping[str, object]:
    dropped: Final = frozenset(additional_drop_params or ())
    return MappingProxyType({key: value for key, value in request_params.items() if key not in dropped})
