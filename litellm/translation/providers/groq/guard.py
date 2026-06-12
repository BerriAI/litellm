"""Raw-shape fidelity guard for the groq chat serializer.

The explicit ``stream: false`` arm (the httpx path keeps the key on the
wire — probed), then the shared openai guard with the FULL message-name
fallback (groq's transform chains into the base GPT one; names ride
verbatim).
"""

from __future__ import annotations

from collections.abc import Mapping

from ...errors import TranslationError
from ..openai_compat.guard import explicit_stream_false
from ..openai_compat.guard import (
    unsupported_request_shapes as openai_unsupported_request_shapes,
)

_Raw = Mapping[str, object]


def unsupported_request_shapes(raw: _Raw) -> TranslationError | None:
    return explicit_stream_false(raw) or openai_unsupported_request_shapes(raw)
