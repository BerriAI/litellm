"""Raw-shape fidelity guard for the deepseek serializer.

The httpx path serializes an explicitly-sent ``stream: false`` onto the wire
(the shared family fact), so that arm runs first; then the shared openai
guard with the FULL message-``name`` fallback — v1's deepseek transform
chains the base ``_transform_messages`` after its flatten and nothing strips
names, so v1 forwards ``name`` verbatim on every role.
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
