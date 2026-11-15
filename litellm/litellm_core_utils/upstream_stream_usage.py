from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.litellm_core_utils.core_helpers import as_str_mapping

UPSTREAM_STREAM_USAGE_KEY: Final = "litellm_upstream_stream_usage"

_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


@dataclass(frozen=True, slots=True)
class AnthropicStreamUsage:
    """The usage objects an Anthropic-wire stream reported, exactly as the provider sent them, so a missing
    or null counter stays distinguishable from a reported zero after normalized usage has filled it in."""

    opening: JsonValue = None
    closing: JsonValue = None
    stop_reason: JsonValue = None


def _as_json(value: object) -> JsonValue:
    try:
        return _JSON.validate_python(value)
    except ValidationError:
        return None


def _field(container: object, key: str) -> object:
    mapping: Final = as_str_mapping(container)
    return mapping.get(key) if mapping is not None else None


def recorded_anthropic_stream_event(
    current: AnthropicStreamUsage | None, chunk: Mapping[str, object]
) -> AnthropicStreamUsage | None:
    match chunk.get("type"):
        case "message_start":
            return AnthropicStreamUsage(opening=_as_json(_field(chunk.get("message"), "usage")))
        case "message_delta":
            return AnthropicStreamUsage(
                opening=current.opening if current is not None else None,
                closing=_as_json(chunk.get("usage")),
                stop_reason=_as_json(_field(chunk.get("delta"), "stop_reason")),
            )
        case _:
            return current


@runtime_checkable
class RecordsUpstreamStreamUsage(Protocol):
    upstream_stream_usage: AnthropicStreamUsage | None
