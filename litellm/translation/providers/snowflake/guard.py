"""Raw-shape fidelity guard for the snowflake serializer.

Just the shared openai guard with the FULL message-``name`` fallback — v1
sends messages VERBATIM (its transform_request never calls super), so
every raw shape the IR cannot round-trip losslessly is a v1-serves
fallback. Deliberately NO explicit-stream-false arm: v1 puts ``stream``
in EVERY body (default false), so absent-vs-false is the same wire byte
and v2 serves both (the one wave-2b provider where that arm would be
wrong).
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
