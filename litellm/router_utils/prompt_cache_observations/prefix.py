import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Final, TypeAlias

from pydantic import JsonValue

from litellm.constants import REDACTED_BY_LITELLM

from ..prompt_caching_cache import PROMPT_CACHE_ONE_HOUR_PIN_TTL_SECONDS, PROMPT_CACHE_PIN_TTL_SECONDS
from .types import (
    CacheablePrefix,
    NoCacheBreakpoint,
    PrefixEvidence,
    PrefixUnassessed,
    UnassessedReason,
    WireShape,
)

_ANTHROPIC_CONTENT_KEYS: Final = frozenset(("tools", "system", "messages", "cache_control"))
_ANTHROPIC_OUTPUT_KEYS: Final = frozenset(
    ("max_tokens", "temperature", "top_p", "top_k", "stop_sequences", "stream", "metadata")
)
_CONVERSE_CONTENT_KEYS: Final = frozenset(("toolConfig", "system", "messages"))
_CONVERSE_OUTPUT_KEYS: Final = frozenset(("inferenceConfig",))
_CACHE_CONTROL: Final = frozenset(("cache_control",))
_CONTENT: Final = frozenset(("content",))
_TOOLS: Final = frozenset(("tools",))
_REDACTION_MARK: Final = json.dumps(REDACTED_BY_LITELLM).encode()


@dataclass(frozen=True, slots=True)
class _Segment:
    identity: JsonValue
    marker: JsonValue


@dataclass(frozen=True, slots=True)
class _Unrecognized:
    pass


_Segments: TypeAlias = tuple[_Segment, ...] | _Unrecognized


def _without(item: Mapping[str, JsonValue], keys: frozenset[str]) -> JsonValue:
    return {key: value for key, value in item.items() if key not in keys}


def _listed(value: JsonValue) -> tuple[JsonValue, ...] | None:
    if value is None:
        return ()
    return tuple(value) if isinstance(value, list) else None


def _flattened(groups: tuple[_Segments, ...]) -> _Segments:
    recognized: Final = tuple(group for group in groups if not isinstance(group, _Unrecognized))
    if len(recognized) != len(groups):
        return _Unrecognized()
    return tuple(chain.from_iterable(recognized))


def _envelope(body: Mapping[str, JsonValue], excluded: frozenset[str], extra: JsonValue = None) -> _Segment:
    return _Segment(identity=["envelope", _without(body, excluded), extra], marker=None)


def _anthropic_segment(kind: str, index: int, position: int, envelope: JsonValue, item: JsonValue) -> _Segment:
    if not isinstance(item, dict):
        return _Segment(identity=[kind, index, envelope, position, item], marker=None)
    return _Segment(
        identity=[kind, index, envelope, position, _without(item, _CACHE_CONTROL)],
        marker=item.get("cache_control"),
    )


def _anthropic_message_segments(index: int, message: JsonValue) -> _Segments:
    if not isinstance(message, dict) or "cache_control" in message:
        return _Unrecognized()
    envelope: Final = _without(message, _CONTENT)
    content: Final = message.get("content")
    match content:
        case str():
            return (_anthropic_segment("message", index, 0, envelope, {"type": "text", "text": content}),)
        case list():
            return tuple(
                _anthropic_segment("message", index, position, envelope, block)
                for position, block in enumerate(content)
            )
        case _:
            return _Unrecognized()


def _anthropic_system(body: Mapping[str, JsonValue]) -> tuple[JsonValue, ...] | None:
    system: Final = body.get("system")
    return (system,) if isinstance(system, str) else _listed(system)


def _anthropic_segments(body: Mapping[str, JsonValue]) -> _Segments:
    tools: Final = _listed(body.get("tools"))
    system: Final = _anthropic_system(body)
    messages: Final = _listed(body.get("messages"))
    if tools is None or system is None or not messages:
        return _Unrecognized()
    return _flattened(
        (
            (_envelope(body, _ANTHROPIC_CONTENT_KEYS | _ANTHROPIC_OUTPUT_KEYS),),
            tuple(_anthropic_segment("tool", index, 0, None, tool) for index, tool in enumerate(tools)),
            tuple(_anthropic_segment("system", index, 0, None, block) for index, block in enumerate(system)),
            *(_anthropic_message_segments(index, message) for index, message in enumerate(messages)),
        )
    )


def _is_cache_point(block: JsonValue) -> bool:
    return isinstance(block, dict) and "cachePoint" in block


def _misplaced_cache_point(blocks: tuple[JsonValue, ...], position: int) -> bool:
    block: Final = blocks[position]
    if not isinstance(block, dict) or not isinstance(block.get("cachePoint"), dict):
        return True
    return len(block) != 1 or position == 0 or _is_cache_point(blocks[position - 1])


def _following_cache_point(blocks: tuple[JsonValue, ...], position: int) -> JsonValue:
    following: Final = blocks[position + 1] if position + 1 < len(blocks) else None
    return following.get("cachePoint") if isinstance(following, dict) and _is_cache_point(following) else None


def _converse_segments_of(kind: str, index: int, envelope: JsonValue, blocks: tuple[JsonValue, ...]) -> _Segments:
    cache_points: Final = tuple(position for position, block in enumerate(blocks) if _is_cache_point(block))
    if any(_misplaced_cache_point(blocks, position) for position in cache_points):
        return _Unrecognized()
    content: Final = tuple(position for position in range(len(blocks)) if position not in cache_points)
    return tuple(
        _Segment(
            identity=[kind, index, envelope, ordinal, blocks[position]],
            marker=_following_cache_point(blocks, position),
        )
        for ordinal, position in enumerate(content)
    )


def _converse_message_segments(index: int, message: JsonValue) -> _Segments:
    if not isinstance(message, dict):
        return _Unrecognized()
    content: Final = _listed(message.get("content"))
    if content is None:
        return _Unrecognized()
    return _converse_segments_of("message", index, _without(message, _CONTENT), content)


def _converse_tool_config(body: Mapping[str, JsonValue]) -> Mapping[str, JsonValue] | _Unrecognized:
    tool_config: Final = body.get("toolConfig")
    if tool_config is None:
        return {}
    return tool_config if isinstance(tool_config, dict) else _Unrecognized()


def _converse_segments(body: Mapping[str, JsonValue]) -> _Segments:
    tool_config: Final = _converse_tool_config(body)
    if isinstance(tool_config, _Unrecognized):
        return _Unrecognized()
    tools: Final = _listed(tool_config.get("tools"))
    system: Final = _listed(body.get("system"))
    messages: Final = _listed(body.get("messages"))
    if tools is None or system is None or not messages:
        return _Unrecognized()
    return _flattened(
        (
            (_envelope(body, _CONVERSE_CONTENT_KEYS | _CONVERSE_OUTPUT_KEYS, _without(tool_config, _TOOLS)),),
            _converse_segments_of("tool", 0, None, tools),
            _converse_segments_of("system", 0, None, system),
            *(_converse_message_segments(index, message) for index, message in enumerate(messages)),
        )
    )


def _ttl_seconds(ttl: JsonValue) -> int | None:
    match ttl:
        case None | "5m":
            return PROMPT_CACHE_PIN_TTL_SECONDS
        case "1h":
            return PROMPT_CACHE_ONE_HOUR_PIN_TTL_SECONDS
        case _:
            return None


def _marker_ttl(wire: WireShape, marker: JsonValue) -> int | UnassessedReason:
    expected_type: Final = "ephemeral" if wire is WireShape.ANTHROPIC_MESSAGES else "default"
    if not isinstance(marker, dict) or marker.get("type") != expected_type:
        return UnassessedReason.BREAKPOINT_UNRECOGNIZED
    ttl: Final = _ttl_seconds(marker.get("ttl"))
    return UnassessedReason.TTL_UNRECOGNIZED if ttl is None else ttl


def _fingerprint(key: bytes, segments: tuple[_Segment, ...]) -> str | None:
    canonical: Final = json.dumps(
        [segment.identity for segment in segments], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    if _REDACTION_MARK in canonical:
        return None
    return hashlib.blake2b(canonical, key=key, digest_size=32).hexdigest()


def outgoing_cacheable_prefix(wire: WireShape, body: Mapping[str, JsonValue], key: bytes) -> PrefixEvidence:
    """The prefix ends at the last breakpoint in tools, system, messages order, or at the end of the prompt
    when Anthropic's top-level cache_control asks the provider to place it. Its TTL is that breakpoint's,
    since an earlier breakpoint caches a different, shorter prefix. Output and sampling settings stay out
    of the identity, since they are not part of the cached prompt."""
    segments: Final = _anthropic_segments(body) if wire is WireShape.ANTHROPIC_MESSAGES else _converse_segments(body)
    if isinstance(segments, _Unrecognized):
        return PrefixUnassessed(UnassessedReason.BREAKPOINT_UNRECOGNIZED)
    marked: Final = tuple(index for index, segment in enumerate(segments) if segment.marker is not None)
    automatic: Final = body.get("cache_control") if wire is WireShape.ANTHROPIC_MESSAGES else None
    if automatic is None and not marked:
        return NoCacheBreakpoint()
    ttls: Final = tuple(_marker_ttl(wire, segments[index].marker) for index in marked) + (
        (_marker_ttl(wire, automatic),) if automatic is not None else ()
    )
    failure: Final = next((ttl for ttl in ttls if isinstance(ttl, UnassessedReason)), None)
    if failure is not None:
        return PrefixUnassessed(failure)
    seconds: Final = tuple(ttl for ttl in ttls if isinstance(ttl, int))
    fingerprint: Final = _fingerprint(key, segments if automatic is not None else segments[: marked[-1] + 1])
    if fingerprint is None:
        return PrefixUnassessed(UnassessedReason.REQUEST_REDACTED)
    return CacheablePrefix(fingerprint=fingerprint, ttl_seconds=seconds[-1], breakpoints=len(ttls))
