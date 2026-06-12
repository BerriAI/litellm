"""Raw-shape fidelity guard for the azure_ai override set.

The azure guard's checks apply unchanged (azure_ai's ``OpenAIConfig``-derived
transform also never strips ``cache_control``, and ``stream`` rides the
httpx body), plus the azure_ai-only flatten: ``AzureAIStudioConfig
._transform_messages`` (azure_ai/chat/transformation.py:174-192) collapses
every list-form content to a joined string unless the list carries an
image_url/input_audio part. The shared inbound parse keeps list-vs-string
form only for the shapes the openai guard already admits, so any text-only
content list falls back to v1's flatten.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from ...errors import TranslationError
from ..azure.guard import unsupported_request_shapes as azure_unsupported_shapes

_Raw = Mapping[str, object]


def unsupported_request_shapes(raw: _Raw) -> TranslationError | None:
    shared = azure_unsupported_shapes(raw)
    if shared is not None:
        return shared
    reason = _flatten_reason(raw)
    if reason is None:
        return None
    return TranslationError.of_unsupported(f"{reason}; v1 forwards the original shape")


def _flatten_reason(raw: _Raw) -> str | None:
    messages = raw.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, str):
        return None
    for item in cast(Sequence[object], messages):
        if not isinstance(item, Mapping):
            continue
        content = cast(Mapping[str, object], item).get("content")
        if not isinstance(content, Sequence) or isinstance(content, str):
            continue
        if not _has_media_part(cast(Sequence[object], content)):
            return (
                "text-only content list (azure_ai flattens it to a string, "
                "azure_ai/chat/transformation.py:174-192)"
            )
    return None


def _has_media_part(content: Sequence[object]) -> bool:
    for part in content:
        if isinstance(part, Mapping):
            part_type = cast(Mapping[str, object], part).get("type")
            if part_type in ("image_url", "input_audio"):
                return True
    return False
