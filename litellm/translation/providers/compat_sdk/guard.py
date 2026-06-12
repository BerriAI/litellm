"""Raw-shape fidelity guard for the SDK-path openai-compat family.

One family-wide arm before the shared openai guard: an explicit
``stream: false``. On this path ``completion()`` forwards the caller's False
into ``get_optional_params`` (non-default against the ``None`` default), it
lands in optional_params, and the SDK serializes the key onto the wire —
while the IR cannot represent absent-vs-false (verified in-process at HEAD;
the same arm the azure and xai guards carry).

The openai guard runs with its full message-``name`` fallback: none of the
family configs strips names (only xai does), so v1 forwards ``name``
verbatim on every role.
"""

from __future__ import annotations

from collections.abc import Mapping

from ...errors import TranslationError
from ..openai_compat.guard import (
    unsupported_request_shapes as openai_unsupported_request_shapes,
)

_Raw = Mapping[str, object]


def unsupported_request_shapes(raw: _Raw) -> TranslationError | None:
    if "stream" in raw and raw.get("stream") is False:
        return TranslationError.of_unsupported(
            "explicit stream: false (the SDK path serializes the key onto "
            "the wire; absent-vs-false is lost in the IR)"
        )
    return openai_unsupported_request_shapes(raw)
