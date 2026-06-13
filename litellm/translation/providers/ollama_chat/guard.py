"""Raw-shape fidelity guard for the ollama_chat NDJSON serializer.

The shared openai guard runs with ``skip_name_fallback`` because v1's message
munge (``OllamaChatCompletionMessage``) whitelists role/thinking/content/
images/tool_calls/tool_call_id — message ``name`` is dropped for EVERY role,
so the IR's name-drop IS v1's behavior (probed in-process at HEAD).

Deliberately NO explicit ``stream: false`` arm: the ollama body ALWAYS
carries ``stream`` (transform_request pops it with default False), so an
explicitly-sent false and an absent key are the same wire byte — the
snowflake shape, pinned by the request gate's ``stream_false`` IDENTICAL row.

The shared arms that fire here are all conservative v1-serves fallbacks
(string stop rides verbatim into ``options.stop``; both max-tokens keys
last-write ``num_predict``; consecutive turns / mid-conversation system
messages ride VERBATIM in v1's munge while the IR merges/hoists them).
"""

from __future__ import annotations

from collections.abc import Mapping

from ...errors import TranslationError
from ..openai_compat.guard import unsupported_request_shapes as _openai_shapes

_Raw = Mapping[str, object]


def unsupported_request_shapes(raw: _Raw) -> TranslationError | None:
    return _openai_shapes(raw, skip_name_fallback=True)
