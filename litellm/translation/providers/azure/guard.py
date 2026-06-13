"""Raw-shape fidelity guard for the azure passthrough serializer.

Runs the openai_compat guard (the shared lossless-round-trip checks) plus the
two azure-only fidelity holes:

- ``cache_control``: azure's transform does NOT strip it (az
  gpt_transformation.py:250-263 has no strip; the openai transform does), so
  v1 forwards the keys verbatim while the IR types them away. Any request
  carrying ``cache_control`` falls back to v1.
- explicit ``stream: false``: azure keeps ``stream`` in optional_params all
  the way into the SDK call (azure.py builds ``{model, messages,
  **optional_params}``), so an explicitly-sent ``false`` reaches the wire;
  absent-vs-false is lost in the IR.
"""

from __future__ import annotations

from collections.abc import Mapping

from ...errors import TranslationError
from ..openai_compat import unsupported_request_shapes as openai_unsupported_shapes
from ..openai_compat.guard import carries_cache_control, explicit_stream_false

_Raw = Mapping[str, object]


def unsupported_request_shapes(raw: _Raw) -> TranslationError | None:
    shared = openai_unsupported_shapes(raw)
    if shared is not None:
        return shared
    stream_false = explicit_stream_false(raw)
    if stream_false is not None:
        return stream_false
    reason = _azure_reason(raw)
    if reason is None:
        return None
    return TranslationError.of_unsupported(f"{reason}; v1 forwards the original shape")


def _azure_reason(raw: _Raw) -> str | None:
    # the recursive scan is openai_compat.guard.carries_cache_control (lifted
    # there at wave-2b-alpha; openrouter composes the same mechanism behind
    # its cache-capable-model policy)
    for field in ("messages", "tools"):
        if carries_cache_control(raw.get(field)):
            return (
                f"cache_control inside {field} (azure forwards it verbatim, "
                "az gpt_transformation.py:250-263; the IR strips it)"
            )
    return None
