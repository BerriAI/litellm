from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import accumulate
from types import MappingProxyType
from typing import Final, cast

from pydantic import JsonValue, TypeAdapter, ValidationError

CACHE_SETTINGS: Final = (
    "system",
    "instructions",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "text",
    "reasoning",
    "reasoning_effort",
    "thinking",
    "verbosity",
    "output_config",
    "output_format",
    "speed",
    "prompt_cache_key",
    "cache_key",
    "cached_content",
    "previous_response_id",
    "conversation",
    "context_management",
    "compaction",
)
BASELINE_PARAMETERS: Final = (
    *CACHE_SETTINGS,
    "prompt_cache_options",
    "prompt_cache_retention",
    "cache_control",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
)
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MAX_BYTES: Final = 4 * 1024 * 1024
_MAX_NODES: Final = 32768
_MAX_DEPTH: Final = 32


def _json_cost(value: object, depth: int = 0) -> Iterator[int]:
    if depth > _MAX_DEPTH:
        yield _MAX_BYTES + 1
    elif isinstance(value, str):
        yield (6 if value.isascii() else 12) * len(value) + 2
    elif isinstance(value, dict):
        yield 2
        for key, item in cast(dict[object, object], value).items():
            yield from _json_cost(key, depth + 1)
            yield from _json_cost(item, depth + 1)
            yield 2
    elif isinstance(value, (list, tuple)):
        yield 2
        for item in cast(list[object] | tuple[object, ...], value):
            yield from _json_cost(item, depth + 1)
            yield 1
    elif isinstance(value, int) and value.bit_length() > 64:
        yield _MAX_BYTES + 1
    elif value is None or isinstance(value, (bool, int, float)):
        yield 32
    else:
        yield _MAX_BYTES + 1


def within_baseline_budget(value: object) -> bool:
    return all(
        size <= _MAX_BYTES and nodes <= _MAX_NODES for nodes, size in enumerate(accumulate(_json_cost(value)), 1)
    )


def _parameters(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    mapping: Final = cast(Mapping[str, object], value)
    return {key: mapping[key] for key in BASELINE_PARAMETERS if mapping.get(key) is not None}


def capture_baseline_parameters(
    kwargs: Mapping[str, object], *, include_extra_body: bool = True
) -> Mapping[str, JsonValue] | None:
    extra: Final = _parameters(kwargs.get("extra_body")) if include_extra_body else {}
    parameters: Final = {**_parameters(kwargs), **({"extra_body": extra} if extra else {})}
    if not within_baseline_budget(parameters):
        return None
    try:
        return MappingProxyType(_JSON_OBJECT.validate_python(parameters))
    except ValidationError:
        return None


def baseline_request(
    kwargs: Mapping[str, object],
    caller: Mapping[str, JsonValue],
    deployment: Mapping[str, object],
    *,
    include_extra_body: bool = True,
) -> Mapping[str, object] | None:
    snapshot: Final = capture_baseline_parameters(deployment)
    if snapshot is None:
        return None
    configured: Final = {
        **_parameters(snapshot),
        **(_parameters(snapshot.get("extra_body")) if include_extra_body else {}),
    }
    requested: Final = {**_parameters(caller), **(_parameters(caller.get("extra_body")) if include_extra_body else {})}
    configured_tools: Final = configured.get("tools")
    caller_tools: Final = requested.get("tools")
    merged_tools: Final = (
        {"tools": [*configured_tools, *caller_tools]}
        if isinstance(configured_tools, list) and isinstance(caller_tools, list)
        else {}
    )
    return MappingProxyType(
        {
            **{key: value for key, value in kwargs.items() if key not in (*BASELINE_PARAMETERS, "extra_body")},
            **{key: value for key, value in configured.items() if value is not None},
            **requested,
            **merged_tools,
        }
    )
