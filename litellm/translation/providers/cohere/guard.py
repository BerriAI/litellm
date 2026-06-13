"""Raw-shape fidelity guard for the cohere v2 chat serializer.

Route predicates first (researcher-4 §11 — provider Literals cannot carry a
route, so the predicate lives HERE, asserted by the differential gates):

- ``"v1/" in model``: the legacy Cohere v1 wire (message/chat_history/
  tool_results) is DON'T-PORT — permanent typed fallback; v1 owns it.
- ``"v2/" in model``: main.py's cohere elif strips the ``v2/`` prefix from
  the wire model BEFORE transform (main.py, cohere branch). That rewrite is
  envelope scope; serving it here would put the prefixed model on the wire.

Then ``explicit_stream_false`` (the httpx path keeps the key on the wire —
v1's map copies ``stream`` verbatim, False included) and the shared openai
guard with the FULL message-name fallback: cohere v2's transform_request is
the inherited OpenAI GPT transform, so message ``name`` is forwarded
verbatim and the IR's name-drop would diverge.
"""

from __future__ import annotations

from collections.abc import Mapping

from ...errors import TranslationError
from ..openai_compat.guard import stream_false_then_unsupported_shapes

_Raw = Mapping[str, object]


def unsupported_request_shapes(raw: _Raw) -> TranslationError | None:
    model = raw.get("model")
    if isinstance(model, str) and "v1/" in model:
        return TranslationError.of_unsupported(
            "cohere v1 route (explicit 'v1/' in the model): the legacy v1 "
            "chat wire is a permanent typed fallback; v1 owns it"
        )
    if isinstance(model, str) and "v2/" in model:
        return TranslationError.of_unsupported(
            "explicit 'v2/' model prefix: v1's cohere completion() branch "
            "strips it before transform (envelope rewrite); v1 owns it"
        )
    # the ONE shared composition after the route predicates
    # (critic-wave2b-alpha NIT-1; sibling-merge sweep)
    return stream_false_then_unsupported_shapes(raw)
