"""Raw-shape fidelity guard for the groq chat serializer.

The explicit ``stream: false`` arm (the httpx path keeps the key on the
wire — probed), then the shared openai guard with the FULL message-name
fallback (groq's transform chains into the base GPT one; names ride
verbatim).
"""

from __future__ import annotations

from collections.abc import Mapping

from ...errors import TranslationError
from ..openai_compat.guard import stream_false_then_unsupported_shapes

_Raw = Mapping[str, object]


def unsupported_request_shapes(raw: _Raw) -> TranslationError | None:
    # the ONE shared composition (critic-wave2b-alpha NIT-1; sibling-merge
    # sweep: this body re-declared it)
    return stream_false_then_unsupported_shapes(raw)
