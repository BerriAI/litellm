"""Raw-shape fidelity guard for the hosted_vllm serializer.

The httpx path serializes an explicitly-sent ``stream: false`` onto the wire
(the shared family fact), then the shared openai guard with the FULL
message-``name`` fallback — v1's hosted_vllm transform chains the base
``_transform_messages`` and nothing strips names. The guard's existing
custom-tool and assistant-``thinking_blocks`` arms double as this provider's
rewrite fallbacks (v1 converts both; pinned in the request gate).
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
