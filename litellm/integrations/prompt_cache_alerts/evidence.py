import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Final, TypeAlias

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.constants import REDACTED_BY_LITELLM

_BODY: Final = TypeAdapter(dict[str, JsonValue])
_CONTENT_KEYS: Final = frozenset(("tools", "system", "messages"))
_OUTPUT_KEYS: Final = frozenset(("max_tokens", "temperature", "top_p", "top_k", "stop_sequences", "stream", "metadata"))
_FIVE_MINUTES: Final = (None, "5m")


@dataclass(frozen=True, slots=True)
class CacheCounters:
    read_tokens: int
    write_tokens: int


@dataclass(frozen=True, slots=True)
class CacheablePrefix:
    fingerprint: str


@dataclass(frozen=True, slots=True)
class NotCaching:
    pass


@dataclass(frozen=True, slots=True)
class Unsupported:
    pass


PrefixEvidence: TypeAlias = CacheablePrefix | NotCaching | Unsupported


def _json_body(raw: object) -> Mapping[str, JsonValue] | None:
    try:
        match raw:
            case str() | bytes():
                return _BODY.validate_json(raw)
            case dict():
                return _BODY.validate_python(raw)
            case _:
                return None
    except ValidationError:
        return None


def _count(value: JsonValue) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def cache_counters(raw_response: object) -> CacheCounters | None:
    """The counters the provider itself reported. A missing, partial, malformed or self-contradicting usage is None,
    never zero, since a normalized response fills an absent counter with 0."""
    body: Final = _json_body(raw_response)
    usage: Final = body.get("usage") if body is not None else None
    stop_reason: Final = body.get("stop_reason") if body is not None else None
    if not isinstance(usage, dict) or not isinstance(stop_reason, str) or not stop_reason:
        return None
    read: Final = _count(usage.get("cache_read_input_tokens"))
    write: Final = _count(usage.get("cache_creation_input_tokens"))
    if read is None or write is None:
        return None
    breakdown: Final = usage.get("cache_creation")
    if breakdown is not None:
        parts: Final = tuple(_count(value) for value in breakdown.values()) if isinstance(breakdown, dict) else (None,)
        if None in parts or sum(part for part in parts if part is not None) != write:
            return None
    return CacheCounters(read_tokens=read, write_tokens=write)


@dataclass(frozen=True, slots=True)
class _Segment:
    identity: JsonValue
    marker: JsonValue


def _blocks(kind: str, index: int, content: JsonValue) -> tuple[_Segment, ...] | None:
    if isinstance(content, str):
        return (_Segment([kind, index, 0, {"type": "text", "text": content}], None),)
    if not isinstance(content, list):
        return None
    return tuple(
        _Segment(
            [kind, index, position, {k: v for k, v in block.items() if k != "cache_control"}],
            block.get("cache_control"),
        )
        if isinstance(block, dict)
        else _Segment([kind, index, position, block], None)
        for position, block in enumerate(content)
    )


def _message_segments(index: int, message: JsonValue) -> tuple[_Segment, ...] | None:
    if not isinstance(message, dict) or "cache_control" in message:
        return None
    blocks: Final = _blocks("message", index, message.get("content"))
    if blocks is None:
        return None
    envelope: Final = _Segment(["message_envelope", index, {k: v for k, v in message.items() if k != "content"}], None)
    return (envelope, *blocks)


def _segments(body: Mapping[str, JsonValue]) -> tuple[_Segment, ...] | None:
    tools: Final = body.get("tools", [])
    system: Final = body.get("system", [])
    messages: Final = body.get("messages")
    if not isinstance(tools, list) or not isinstance(messages, list) or not messages:
        return None
    groups: Final = (
        _blocks("tool", 0, tools),
        _blocks("system", 0, system),
        *(_message_segments(index, message) for index, message in enumerate(messages)),
    )
    if any(group is None for group in groups):
        return None
    envelope: Final = {k: v for k, v in body.items() if k not in _CONTENT_KEYS | _OUTPUT_KEYS}
    return (
        _Segment(["envelope", envelope], None),
        *chain.from_iterable(group for group in groups if group is not None),
    )


def _fingerprint(key: bytes, segments: Sequence[_Segment]) -> str | None:
    canonical: Final = json.dumps(
        [segment.identity for segment in segments], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    if REDACTED_BY_LITELLM in canonical:
        return None
    return hashlib.blake2b(canonical.encode(), key=key, digest_size=32).hexdigest()


def cacheable_prefix(body: object, key: bytes) -> PrefixEvidence:
    """Only an Anthropic request with exactly one explicit five-minute breakpoint has a prefix. Any other request
    that asks for caching, or whose body cannot be compared, is unsupported. The prefix runs through the marked
    block in tools, system, messages order, and output and sampling settings are left out."""
    parsed: Final = _json_body(body)
    if parsed is None:
        return Unsupported()
    segments: Final = _segments(parsed)
    if segments is None or parsed.get("cache_control") is not None:
        return Unsupported()
    marked: Final = tuple(index for index, segment in enumerate(segments) if segment.marker is not None)
    if not marked:
        return NotCaching()
    marker: Final = segments[marked[0]].marker
    five_minute: Final = (
        isinstance(marker, dict) and marker.get("type") == "ephemeral" and marker.get("ttl") in _FIVE_MINUTES
    )
    if len(marked) != 1 or not five_minute:
        return Unsupported()
    fingerprint: Final = _fingerprint(key, segments[: marked[0] + 1])
    return Unsupported() if fingerprint is None else CacheablePrefix(fingerprint)


def keyed_digest(key: bytes, *parts: str | None) -> str:
    return hashlib.blake2b(json.dumps(parts).encode(), key=key, digest_size=32).hexdigest()


def fingerprint_key(salt: str) -> bytes:
    return hashlib.blake2b(salt.encode(), digest_size=32, person=b"prompt-cache").digest()
