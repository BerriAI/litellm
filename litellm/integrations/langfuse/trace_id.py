"""Trace id normalization shared by the ``langfuse`` and ``langfuse_otel`` callbacks, importable without the Langfuse SDK."""

from __future__ import annotations

import re
from hashlib import sha256
from typing import Final

from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

_TRACE_ID_PATTERN: Final = re.compile(r"^(?=.*[1-9a-f])[0-9a-f]{32}$")


def resolve_trace_id(trace_id: object | None) -> str:
    """Map a caller's trace id onto the 32 lowercase hex characters v4 requires."""
    serialized: Final = "" if trace_id is None else str(trace_id)
    normalized: Final = serialized.lower().replace("-", "")
    if _TRACE_ID_PATTERN.fullmatch(normalized):
        return normalized
    if not serialized:
        return format(RandomIdGenerator().generate_trace_id(), "032x")
    return sha256(serialized.encode("utf-8")).digest()[:16].hex()
