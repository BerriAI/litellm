"""Raw-shape fidelity guard for the SDK-path openai-compat family.

The shared explicit ``stream: false`` arm runs first: on this path
``completion()`` forwards the caller's False into ``get_optional_params``
(non-default against the ``None`` default), it lands in optional_params,
and the SDK serializes the key onto the wire — and the family's httpx
member (cometapi) keeps the key on the wire the same way, so the arm
covers both paths — while the IR cannot represent absent-vs-false
(verified in-process at HEAD for both; the same arm the azure and xai
guards compose).

The openai guard then runs with its full message-``name`` fallback: none of
the family configs strips names (only xai does), so v1 forwards ``name``
verbatim on every role.
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
