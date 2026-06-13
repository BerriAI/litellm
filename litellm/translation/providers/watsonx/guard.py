"""Raw-shape fidelity guard for the watsonx chat serializer.

The shared openai guard with the FULL message-``name`` fallback: watsonx's
transform is the inherited base ``_transform_messages`` (the openai_like
handler calls it directly), so names ride the wire verbatim.

Deliberately NO explicit ``stream: false`` arm: the openai_like handler
pops ``stream`` and re-adds it UNCONDITIONALLY (``stream or False``), so
the wire body carries ``stream: false`` whether the caller sent False or
nothing — the v2 serializer emits the key on every request, and the IR's
absent-vs-false collapse is invisible on this wire (probed in-process).

Auth is pure envelope (Authorization passthrough -> token/ZenApiKey -> the
``generate_iam_token`` NETWORK POST with its in-memory cache, all inside
``validate_environment``): the v1 resolution stays at the seam; nothing
here touches it (semgrep-enforced — no ambient I/O outside deps/engine).
"""

from __future__ import annotations

from collections.abc import Mapping

from ...errors import TranslationError
from ..openai_compat.guard import (
    unsupported_request_shapes as openai_unsupported_request_shapes,
)

_Raw = Mapping[str, object]


def unsupported_request_shapes(raw: _Raw) -> TranslationError | None:
    return openai_unsupported_request_shapes(raw)
