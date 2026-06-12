"""Raw-shape fidelity guard for the mistral chat serializer.

Mistral's name semantics are a per-role/per-branch matrix, so this guard
runs its OWN name arm and composes the shared openai guard with
``skip_name_fallback`` (probed in-process at HEAD):

- the non-image transform branch strips ``name`` from every non-tool role
  (and from tool messages whose name strip-blanks) — the IR's name-drop IS
  v1 there, so those serve;
- a TOOL-role message with a non-blank ``name`` keeps it on the wire — the
  IR drops it, fall back;
- the IMAGE/FILE branch returns base-transformed messages with EVERY name
  forwarded verbatim — any name beside image/file content falls back.

There is NO explicit ``stream: false`` arm: mistral's map only copies
``stream`` when the value is True (probed: explicit False never reaches
optional_params), so the IR's absent-vs-false collapse IS v1's drop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from ...errors import TranslationError
from ..openai_compat.guard import (
    unsupported_request_shapes as openai_unsupported_request_shapes,
)

_Raw = Mapping[str, object]


def unsupported_request_shapes(raw: _Raw) -> TranslationError | None:
    reason = _name_reason(raw) or _malformed_entry_reason(raw)
    if reason is not None:
        return TranslationError.of_unsupported(reason)
    return openai_unsupported_request_shapes(raw, skip_name_fallback=True)


def _malformed_entry_reason(raw: _Raw) -> str | None:
    """verifier-wave2b-beta F10: the shared guard's 'v1 forwards the
    original shape' suffix is FALSE for mistral's non-object content-list
    entries — v1's ``_transform_messages`` fork check reads ``part.get``
    over the raw list and raises AttributeError. Fallback-safe either way;
    this arm fires first so the reason names mistral's real v1 path."""
    messages = _as_seq(raw.get("messages"))
    if messages is None:
        return None
    for item in messages:
        entry = _as_map(item)
        if entry is None:
            continue
        content = _as_seq(entry.get("content"))
        if content is None:
            continue
        if any(_as_map(part) is None for part in content):
            return (
                "non-object content-list entry: v1's mistral "
                "_transform_messages fork check raises AttributeError on it"
            )
    return None


def _name_reason(raw: _Raw) -> str | None:
    messages = _as_seq(raw.get("messages"))
    if messages is None:
        return None
    entries = [entry for item in messages if (entry := _as_map(item)) is not None]
    has_media = any(_carries_media(entry) for entry in entries)
    for entry in entries:
        name = entry.get("name")
        if name is None:
            continue
        if has_media:
            return (
                "message name beside image/file content: v1's mistral image "
                "branch forwards every name verbatim; v1 serves it"
            )
        if entry.get("role") == "tool" and not _blank_string(name):
            return (
                "tool-message name: v1 keeps it on the wire (the only role "
                "whose name survives); the IR drops it, v1 serves"
            )
    return None


def _carries_media(entry: _Raw) -> bool:
    content = _as_seq(entry.get("content"))
    if content is None:
        return False
    return any(
        (part := _as_map(item)) is not None
        and part.get("type") in ("image_url", "file")
        for item in content
    )


def _blank_string(name: object) -> bool:
    return isinstance(name, str) and len(name.strip()) == 0


def _as_map(value: object) -> _Raw | None:
    if isinstance(value, Mapping):
        return cast(_Raw, value)
    return None


def _as_seq(value: object) -> Sequence[object] | None:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return cast(Sequence[object], value)
    return None
