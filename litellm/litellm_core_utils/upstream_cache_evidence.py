"""Per-attempt evidence the prompt cache observer reads from a call's logging details.

Normalized usage fills an absent cache counter with zero and the outgoing request can be replaced after it is
sent, so these records keep what the provider and the runtime actually did. They hold counters, stop reasons
and settings only, never prompt content, and are never added to a response.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import reduce
from typing import Final, Protocol, TypeAlias, runtime_checkable

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.litellm_core_utils.core_helpers import as_str_mapping

SERVED_RUNTIME_KEY: Final = "litellm_served_runtime"
UPSTREAM_STREAM_USAGE_KEY: Final = "litellm_upstream_stream_usage"
DEPLOYMENT_CACHE_SETTINGS_KEY: Final = "litellm_deployment_cache_settings"
TERMINAL_EVENT_WAIT_SECONDS: Final = 1.0
TERMINAL_EVENT_MAX_LINES: Final = 16

_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_EVENT: Final[TypeAdapter[Mapping[str, JsonValue]]] = TypeAdapter(Mapping[str, JsonValue])
_ANTHROPIC_LIFECYCLE: Final = ("message_start", "message_delta", "message_stop")
_ANTHROPIC_CACHE_COUNTERS: Final = ("cache_read_input_tokens", "cache_creation_input_tokens")
_CONVERSE_TERMINAL: Final = ('"stopReason"', '"usage"')


@dataclass(frozen=True, slots=True)
class AnthropicStreamUsage:
    """The usage an Anthropic-wire stream reported, as sent: `message_start`, the last `message_delta`, and the
    usage some providers repeat on `message_stop`. `stopped` is only set by `message_stop`, since a stream cut
    after its final `message_delta` already carries a stop reason without having completed."""

    opening: JsonValue = None
    closing: JsonValue = None
    final: JsonValue = None
    stop_reason: JsonValue = None
    starts: int = 0
    stopped: bool = False
    out_of_order: bool = False
    conflicting: bool = False


@dataclass(frozen=True, slots=True)
class ConverseStreamUsage:
    """The Converse `messageStop` reason and `metadata` usage, as sent. Converse closes a stream with exactly one
    of each, the stop first."""

    stop_reason: JsonValue = None
    usage: JsonValue = None
    stops: int = 0
    usage_events: int = 0
    out_of_order: bool = False


UpstreamStreamUsage: TypeAlias = AnthropicStreamUsage | ConverseStreamUsage


@runtime_checkable
class RecordsUpstreamStreamUsage(Protocol):
    upstream_stream_usage: UpstreamStreamUsage | None


@runtime_checkable
class DrainsTerminalFrames(Protocol):
    @property
    def awaits_terminal_event(self) -> bool: ...

    async def adrain_terminal_frames(self, max_lines: int) -> None: ...


@runtime_checkable
class RecordsDeploymentCacheSettings(Protocol):
    def record_deployment_cache_settings(self, settings: object) -> None: ...


def _as_json(value: object) -> JsonValue:
    try:
        return _JSON.validate_python(value)
    except ValidationError:
        return None


def _field(container: object, key: str) -> object:
    mapping: Final = as_str_mapping(container)
    return mapping.get(key) if mapping is not None else None


def _cache_counters(usage: JsonValue) -> Mapping[str, JsonValue]:
    reported: Final = as_str_mapping(usage) or {}
    return {key: _as_json(reported[key]) for key in _ANTHROPIC_CACHE_COUNTERS if reported.get(key) is not None}


def _disagrees(earlier: JsonValue, later: JsonValue) -> bool:
    before: Final = _cache_counters(earlier)
    after: Final = _cache_counters(later)
    return any(key in before and before[key] != value for key, value in after.items())


def recorded_anthropic_stream_event(
    current: AnthropicStreamUsage | None, event: Mapping[str, object]
) -> AnthropicStreamUsage | None:
    state: Final = current or AnthropicStreamUsage()
    event_type: Final = event.get("type")
    if state.stopped and event_type != "ping":
        return replace(state, out_of_order=True)
    match event_type:
        case "message_start":
            return replace(
                state,
                opening=_as_json(_field(event.get("message"), "usage")),
                starts=state.starts + 1,
                out_of_order=state.out_of_order or state.closing is not None,
            )
        case "message_delta":
            closing: Final = _as_json(event.get("usage"))
            return replace(
                state,
                closing=closing,
                stop_reason=_as_json(_field(event.get("delta"), "stop_reason")),
                out_of_order=state.out_of_order or state.starts == 0,
                conflicting=state.conflicting or _disagrees(state.closing, closing),
            )
        case "message_stop":
            final: Final = _as_json(event.get("usage"))
            return replace(
                state,
                final=final,
                stopped=True,
                out_of_order=state.out_of_order or state.closing is None,
                conflicting=state.conflicting or _disagrees(state.closing, final) or _disagrees(state.opening, final),
            )
        case _:
            return current


def _parsed_event(text: str) -> Mapping[str, JsonValue] | None:
    try:
        return _EVENT.validate_json(text)
    except ValidationError:
        return None


def recorded_anthropic_stream_text(current: AnthropicStreamUsage | None, text: str) -> AnthropicStreamUsage | None:
    if not any(name in text for name in _ANTHROPIC_LIFECYCLE):
        return current
    event: Final = _parsed_event(text)
    return current if event is None else recorded_anthropic_stream_event(current, event)


def recorded_converse_stream_text(current: ConverseStreamUsage | None, text: str) -> ConverseStreamUsage | None:
    if not any(name in text for name in _CONVERSE_TERMINAL):
        return current
    event: Final = _parsed_event(text)
    if event is None:
        return current
    state: Final = current or ConverseStreamUsage()
    if "stopReason" in event:
        return replace(
            state,
            stop_reason=event["stopReason"],
            stops=state.stops + 1,
            out_of_order=state.out_of_order or state.usage_events > 0,
        )
    if "usage" in event:
        return replace(state, usage=event["usage"], usage_events=state.usage_events + 1)
    return current


def anthropic_stream_usage_from_sse(frames: bytes) -> AnthropicStreamUsage | None:
    """The record for a Messages stream from its complete SSE frames, which the Messages paths keep as the
    bytes the provider sent."""
    events: Final = (
        line[len(b"data:") :].strip().decode("utf-8", errors="replace")
        for line in frames.splitlines()
        if line.startswith(b"data:")
    )
    return reduce(recorded_anthropic_stream_text, events, None)
