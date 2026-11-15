from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.litellm_core_utils.upstream_cache_evidence import AnthropicStreamUsage, ConverseStreamUsage

from .types import CacheCounters, UpstreamCacheUsage, UsageUnknown, UsageUnknownReason, WireShape

_BODY: Final = TypeAdapter(dict[str, JsonValue])


def _counter(usage: Mapping[str, JsonValue], key: str) -> int | UsageUnknownReason:
    value: Final = usage.get(key)
    if value is None:
        return UsageUnknownReason.MISSING
    return value if isinstance(value, int) and _is_count(value) else UsageUnknownReason.MALFORMED


def _counters(usage: Mapping[str, JsonValue], read_key: str, write_key: str) -> UpstreamCacheUsage:
    match _counter(usage, read_key), _counter(usage, write_key):
        case int() as read, int() as write:
            return CacheCounters(read_tokens=read, write_tokens=write)
        case (UsageUnknownReason.MALFORMED, _) | (_, UsageUnknownReason.MALFORMED):
            return UsageUnknown(UsageUnknownReason.MALFORMED)
        case UsageUnknownReason.MISSING, UsageUnknownReason.MISSING:
            return UsageUnknown(UsageUnknownReason.MISSING)
        case _:
            return UsageUnknown(UsageUnknownReason.PARTIAL)


@dataclass(frozen=True, slots=True)
class _Breakdown:
    total: int
    complete: bool


def _is_count(value: JsonValue) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _breakdown(item: JsonValue, keys: tuple[str, ...]) -> _Breakdown | UsageUnknownReason:
    if not isinstance(item, dict):
        return UsageUnknownReason.MALFORMED
    present: Final = tuple(item[key] for key in keys if item.get(key) is not None)
    counts: Final = tuple(value for value in present if isinstance(value, int) and _is_count(value))
    if len(counts) != len(present):
        return UsageUnknownReason.MALFORMED
    return _Breakdown(total=sum(counts), complete=len(counts) == len(keys))


def _checked_against(
    usage: UpstreamCacheUsage, breakdown: _Breakdown | UsageUnknownReason | None
) -> UpstreamCacheUsage:
    """A breakdown only ever confirms the top-level write counter. A partial one cannot stand in for the keys
    it leaves out, so it is only held to not exceeding the total."""
    match usage, breakdown:
        case CacheCounters(), UsageUnknownReason():
            return UsageUnknown(UsageUnknownReason.CONTRADICTORY)
        case CacheCounters(write_tokens=write), _Breakdown(total=total, complete=True) if total != write:
            return UsageUnknown(UsageUnknownReason.CONTRADICTORY)
        case CacheCounters(write_tokens=write), _Breakdown(total=total, complete=False) if total > write:
            return UsageUnknown(UsageUnknownReason.CONTRADICTORY)
        case _:
            return usage


def _anthropic_usage(body: Mapping[str, JsonValue]) -> UpstreamCacheUsage:
    stop_reason: Final = body.get("stop_reason")
    if not isinstance(stop_reason, str) or not stop_reason:
        return UsageUnknown(UsageUnknownReason.INCOMPLETE)
    usage: Final = body.get("usage")
    if not isinstance(usage, dict):
        return UsageUnknown(UsageUnknownReason.MISSING)
    if usage.get("iterations"):
        return UsageUnknown(UsageUnknownReason.ITERATED)
    breakdown: Final = usage.get("cache_creation")
    return _checked_against(
        _counters(usage, "cache_read_input_tokens", "cache_creation_input_tokens"),
        None
        if breakdown is None
        else _breakdown(breakdown, ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")),
    )


def _converse_details_total(details: JsonValue) -> _Breakdown | UsageUnknownReason | None:
    if details is None:
        return None
    if not isinstance(details, list):
        return UsageUnknownReason.MALFORMED
    totals: Final = tuple(_breakdown(detail, ("inputTokens",)) for detail in details)
    complete: Final = tuple(total.total for total in totals if isinstance(total, _Breakdown) and total.complete)
    return (
        _Breakdown(total=sum(complete), complete=True) if len(complete) == len(totals) else UsageUnknownReason.MALFORMED
    )


def _converse_usage(body: Mapping[str, JsonValue]) -> UpstreamCacheUsage:
    stop_reason: Final = body.get("stopReason")
    if not isinstance(stop_reason, str) or not stop_reason:
        return UsageUnknown(UsageUnknownReason.INCOMPLETE)
    usage: Final = body.get("usage")
    if not isinstance(usage, dict):
        return UsageUnknown(UsageUnknownReason.MISSING)
    return _checked_against(
        _counters(usage, "cacheReadInputTokens", "cacheWriteInputTokens"),
        _converse_details_total(usage.get("cacheDetails")),
    )


def _response_body(raw: object) -> Mapping[str, JsonValue] | UsageUnknown:
    try:
        match raw:
            case str() | bytes():
                return _BODY.validate_json(raw)
            case dict():
                return _BODY.validate_python(raw)
            case _:
                return UsageUnknown(UsageUnknownReason.BODY_UNAVAILABLE)
    except ValidationError:
        return UsageUnknown(UsageUnknownReason.BODY_MALFORMED)


def upstream_cache_usage(wire: WireShape, raw_response: object) -> UpstreamCacheUsage:
    body: Final = _response_body(raw_response)
    if isinstance(body, UsageUnknown):
        return body
    match wire:
        case WireShape.ANTHROPIC_MESSAGES:
            return _anthropic_usage(body)
        case WireShape.BEDROCK_CONVERSE:
            return _converse_usage(body)


def _reported(usage: JsonValue) -> Mapping[str, JsonValue]:
    return {key: value for key, value in usage.items() if value is not None} if isinstance(usage, dict) else {}


def _anthropic_stream_usage(stream: AnthropicStreamUsage) -> UpstreamCacheUsage:
    """Only a stream that opened once, closed with `message_stop` after its final `message_delta` and reported
    nothing out of order counts. A counter comes from the last event that reported it, and two events that
    disagree make the usage contradictory."""
    if stream.starts != 1 or not stream.stopped or stream.errored:
        return UsageUnknown(UsageUnknownReason.INCOMPLETE)
    if stream.out_of_order:
        return UsageUnknown(UsageUnknownReason.MALFORMED)
    if stream.conflicting:
        return UsageUnknown(UsageUnknownReason.CONTRADICTORY)
    merged: Final = {**_reported(stream.opening), **_reported(stream.closing), **_reported(stream.final)}
    return _anthropic_usage({"stop_reason": stream.stop_reason, "usage": merged})


def _converse_stream_usage(stream: ConverseStreamUsage) -> UpstreamCacheUsage:
    if stream.stops != 1 or stream.usage_events != 1:
        return UsageUnknown(UsageUnknownReason.INCOMPLETE)
    if stream.out_of_order:
        return UsageUnknown(UsageUnknownReason.MALFORMED)
    return _converse_usage({"stopReason": stream.stop_reason, "usage": stream.usage})


def stream_cache_usage(wire: WireShape, stream: object) -> UpstreamCacheUsage:
    """The record must match the wire the request went out on: a Converse request answered by an Anthropic-shaped
    record means the stream was translated, and a translated stream carries synthesized counters."""
    match wire, stream:
        case WireShape.ANTHROPIC_MESSAGES, AnthropicStreamUsage():
            return _anthropic_stream_usage(stream)
        case WireShape.BEDROCK_CONVERSE, ConverseStreamUsage():
            return _converse_stream_usage(stream)
        case _:
            return UsageUnknown(UsageUnknownReason.STREAM_USAGE_NOT_CAPTURED)
