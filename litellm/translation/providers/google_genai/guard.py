"""Raw-shape guard for the google routes (vertex_ai gemini + AI Studio).

The cache-marker gate (serialize.py) proves v1's context-cache network call
unreachable by bounding v1's token count from the IR — but the inbound parse
drops the OpenAI message ``name`` field (no IR field; v1's generateContent
transform ignores it on the wire), so ``name`` bytes are INVISIBLE to the
bound while v1's ``is_prompt_caching_valid_prompt`` token-counts them
(openai_token_counter charges for ``name``). A request carrying BOTH a cache
marker and any message ``name`` therefore falls back to v1 before parse
(verifier-integration blocker): the bytes cannot be bounded post-IR.

Runs over the UNTRUSTED raw body BEFORE parse; every check is structural and
conservative — the guard can only widen the fallback surface, never change a
served body. vertex_anthropic needs no row here: v1's context caching is
gemini-only (check_and_create_cache lives on the generateContent path) and
the anthropic wire ignores ``name`` on both sides.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

from ...errors import TranslationError

_Raw = Mapping[str, object]


def unsupported_request_shapes(raw: _Raw) -> TranslationError | None:
    messages = raw.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, str):
        return None  # parse rejects malformed messages with its own error
    entries = cast(Sequence[object], messages)
    has_name = any(isinstance(entry, Mapping) and "name" in entry for entry in entries)
    if not has_name:
        return None
    if _carries_cache_control(entries, 0):
        return TranslationError.of_unsupported(
            "message 'name' beside cache_control markers: the IR drops 'name'"
            " so its bytes are invisible to the cache-marker token bound,"
            " while v1's check_and_create_cache token-counts them"
        )
    return None


def _carries_cache_control(value: object, depth: int) -> bool:
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        # exhaustion must never ADMIT a request: treat the unscannable tail
        # as if it carried the marker (fall back to v1)
        return True
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        if "cache_control" in mapping:
            return True
        return any(_carries_cache_control(item, depth + 1) for item in mapping.values())
    if isinstance(value, Sequence) and not isinstance(value, str):
        return any(
            _carries_cache_control(item, depth + 1)
            for item in cast(Sequence[object], value)
        )
    return False
