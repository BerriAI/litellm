"""Raw-shape fidelity guard for the openrouter serializer.

The httpx path serializes an explicitly-sent ``stream: false`` onto the wire
(the shared family fact), so that arm runs first. Then the cache-capable arm:
for models matching v1's CacheControlSupportedModels substrings, v1 KEEPS
``cache_control`` on the wire and MOVES message-level markers into the last
content block (``_move_cache_control_to_content``) — any marker on those
models falls back. Every OTHER model gets the base recursive strip, which
the IR's silent drop reproduces byte-identically, so markers there SERVE
(pinned by the cache_control_stripped row). Last, the shared openai guard
with the FULL message-``name`` fallback — nothing on the openrouter chain
strips names.
"""

from __future__ import annotations

from collections.abc import Mapping

from ...errors import TranslationError
from ..openai_compat.guard import (
    carries_cache_control,
    explicit_stream_false,
)
from ..openai_compat.guard import (
    unsupported_request_shapes as openai_unsupported_request_shapes,
)
from .params import supports_cache_control_in_content

_Raw = Mapping[str, object]


def unsupported_request_shapes(raw: _Raw) -> TranslationError | None:
    return (
        explicit_stream_false(raw)
        or _cache_capable_model_reason(raw)
        or openai_unsupported_request_shapes(raw)
    )


def _cache_capable_model_reason(raw: _Raw) -> TranslationError | None:
    model = raw.get("model")
    if not isinstance(model, str) or not supports_cache_control_in_content(model):
        return None
    for field in ("messages", "tools"):
        if carries_cache_control(raw.get(field)):
            return TranslationError.of_unsupported(
                f"cache_control inside {field} on a cache-capable openrouter "
                "model: v1 keeps it on the wire and moves message-level "
                "markers into the last content block "
                "(_move_cache_control_to_content); v1 serves that rewrite"
            )
    return None
