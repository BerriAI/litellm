from collections.abc import Mapping
from typing import Final, TypeAlias

from pydantic import JsonValue, TypeAdapter, ValidationError

from .types import CacheCounters, UpstreamCacheUsage, UsageUnknown, UsageUnknownReason, WireShape

_BODY: Final = TypeAdapter(dict[str, JsonValue])


def _counter(usage: Mapping[str, JsonValue], key: str) -> int | UsageUnknownReason:
    value: Final = usage.get(key)
    if value is None:
        return UsageUnknownReason.MISSING
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return UsageUnknownReason.MALFORMED
    return value


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


_Breakdown: TypeAlias = int | None | UsageUnknownReason


def _breakdown_total(item: JsonValue, keys: tuple[str, ...]) -> _Breakdown:
    if not isinstance(item, dict):
        return UsageUnknownReason.MALFORMED
    values: Final = tuple(item.get(key, 0) for key in keys)
    counted: Final = tuple(value for value in values if isinstance(value, int) and not isinstance(value, bool))
    return sum(counted) if len(counted) == len(values) else UsageUnknownReason.MALFORMED


def _checked_against(usage: UpstreamCacheUsage, breakdown: _Breakdown) -> UpstreamCacheUsage:
    match usage, breakdown:
        case CacheCounters(), UsageUnknownReason():
            return UsageUnknown(UsageUnknownReason.CONTRADICTORY)
        case CacheCounters(write_tokens=write), int() as total if total != write:
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
        else _breakdown_total(breakdown, ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")),
    )


def _converse_details_total(details: JsonValue) -> _Breakdown:
    if details is None:
        return None
    if not isinstance(details, list):
        return UsageUnknownReason.MALFORMED
    totals: Final = tuple(_breakdown_total(detail, ("inputTokens",)) for detail in details)
    counted: Final = tuple(total for total in totals if isinstance(total, int))
    return sum(counted) if len(counted) == len(totals) else UsageUnknownReason.MALFORMED


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
