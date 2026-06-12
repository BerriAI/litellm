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

from collections.abc import Mapping, Sequence
from typing import cast

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

from ...errors import TranslationError
from ..openai_compat import unsupported_request_shapes as openai_unsupported_shapes

_Raw = Mapping[str, object]


def unsupported_request_shapes(raw: _Raw) -> TranslationError | None:
    shared = openai_unsupported_shapes(raw)
    if shared is not None:
        return shared
    reason = _azure_reason(raw)
    if reason is None:
        return None
    return TranslationError.of_unsupported(f"{reason}; v1 forwards the original shape")


def _azure_reason(raw: _Raw) -> str | None:
    if "stream" in raw and raw.get("stream") is False:
        return "explicit stream: false (azure keeps the key on the wire; absent-vs-false is lost in the IR)"
    for field in ("messages", "tools"):
        if _carries_cache_control(raw.get(field), 0):
            return (
                f"cache_control inside {field} (azure forwards it verbatim, "
                "az gpt_transformation.py:250-263; the IR strips it)"
            )
    return None


def _carries_cache_control(value: object, depth: int) -> bool:
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        return False
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        if "cache_control" in mapping:
            return True
        return any(_carries_cache_control(item, depth + 1) for item in mapping.values())
    if isinstance(value, Sequence) and not isinstance(value, str):
        return any(
            _carries_cache_control(item, depth + 1)
            for item in cast(Sequence[object], value)
        )
    return False
