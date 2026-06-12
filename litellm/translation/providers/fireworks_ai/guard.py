"""Raw-shape fidelity guard for the fireworks_ai serializer.

The httpx path serializes an explicitly-sent ``stream: false`` onto the
wire (the shared family fact), then the shared openai guard with the FULL
message-``name`` fallback — nothing on the fireworks chain strips names.
The guard's existing custom-tool / assistant-``thinking_blocks`` /
``provider_specific_fields`` arms double as this provider's rewrite
fallbacks (v1 pops or converts them; pinned in the request gate), and
``file`` parts fall back at the inbound boundary (v1 migrates pdf files to
``image_url``). ``cache_control`` needs NO arm: v1 strips it recursively
(``filter_value_from_dict``) — the IR drop IS v1, served and pinned.
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
