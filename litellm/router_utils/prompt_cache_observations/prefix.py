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
)

_ANTHROPIC_CONTENT_KEYS: Final = frozenset(("tools", "system", "messages", "cache_control"))
_ANTHROPIC_OUTPUT_KEYS: Final = frozenset(
    ("max_tokens", "temperature", "top_p", "top_k", "stop_sequences", "stream", "metadata")
)
_CACHE_CONTROL: Final = frozenset(("cache_control",))
_CONTENT: Final = frozenset(("content",))
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


def _envelope(body: Mapping[str, JsonValue]) -> _Segment:
    return _Segment(
        identity=["envelope", _without(body, _ANTHROPIC_CONTENT_KEYS | _ANTHROPIC_OUTPUT_KEYS)], marker=None
    )


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
            (_envelope(body),),
            tuple(_anthropic_segment("tool", index, 0, None, tool) for index, tool in enumerate(tools)),
            tuple(_anthropic_segment("system", index, 0, None, block) for index, block in enumerate(system)),
            *(_anthropic_message_segments(index, message) for index, message in enumerate(messages)),
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


def _marker_ttl(marker: JsonValue) -> int | UnassessedReason:
    if not isinstance(marker, dict) or marker.get("type") != "ephemeral":
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


def outgoing_cacheable_prefix(body: Mapping[str, JsonValue], key: bytes) -> PrefixEvidence:
    """The prefix ends at the last breakpoint in tools, system, messages order, or at the end of the prompt
    when Anthropic's top-level cache_control asks the provider to place it. Its TTL is that breakpoint's,
    since an earlier breakpoint caches a different, shorter prefix. Output and sampling settings stay out
    of the identity, since they are not part of the cached prompt."""
    segments: Final = _anthropic_segments(body)
    if isinstance(segments, _Unrecognized):
        return PrefixUnassessed(UnassessedReason.BREAKPOINT_UNRECOGNIZED)
    marked: Final = tuple(index for index, segment in enumerate(segments) if segment.marker is not None)
    automatic: Final = body.get("cache_control")
    if automatic is None and not marked:
        return NoCacheBreakpoint()
    ttls: Final = tuple(_marker_ttl(segments[index].marker) for index in marked) + (
        (_marker_ttl(automatic),) if automatic is not None else ()
    )
    failure: Final = next((ttl for ttl in ttls if isinstance(ttl, UnassessedReason)), None)
    if failure is not None:
        return PrefixUnassessed(failure)
    seconds: Final = tuple(ttl for ttl in ttls if isinstance(ttl, int))
    fingerprint: Final = _fingerprint(key, segments if automatic is not None else segments[: marked[-1] + 1])
    if fingerprint is None:
        return PrefixUnassessed(UnassessedReason.REQUEST_REDACTED)
    return CacheablePrefix(fingerprint=fingerprint, ttl_seconds=seconds[-1], breakpoints=len(ttls))
