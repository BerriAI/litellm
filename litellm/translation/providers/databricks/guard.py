"""Raw-shape fidelity guard for the databricks serializer.

databricks' wire is openai-chat-shaped (``transform_request`` is
OpenAIGPTConfig's), but ``_transform_messages`` runs a databricks-specific
munge BEFORE the base assembly (probed in-process at HEAD):

- message ``name`` is stripped for ASSISTANT and TOOL roles but KEPT for USER
  (``strip_name_from_message(allowed_name_roles=["user"])``). So the IR's
  name-drop matches v1 only off the user role -> the ``name_fallback_user_only``
  arm (the xai/mistral shape): a USER ``name`` falls back (v1 forwards it, the
  IR drops it), an assistant/tool ``name`` SERVES (v1 strips it == the IR drop).
- message-level ``cache_control`` is MOVED into a text block, and a
  whitespace-only message LOSES the marker together with the sanitized block
  (probed) — a lossy interaction the IR cannot reproduce. ANY ``cache_control``
  anywhere in messages or tools falls back so v1 serves its own move/drop (the
  azure verbatim-forward precedent over the shared ``carries_cache_control``
  scanner).
- ``_sanitize_empty_content`` REMOVES the ``content`` key for empty or
  whitespace-only string content (``"   "`` -> ``{"role": "user"}``, probed),
  which the IR keeps as a text block. The openai guard only catches
  ``content is None``, so this guard adds the whitespace-only string arm and
  falls back (v1 serves the bare ``{"role": ...}``).

NO explicit ``stream: false`` arm: databricks ALWAYS materializes ``stream``
(default False in optional_params), so absent and explicit-false are the same
wire byte (the snowflake shape, pinned by the request gate's IDENTICAL row).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from ...errors import TranslationError
from ..openai_compat.guard import (
    carries_cache_control,
)
from ..openai_compat.guard import (
    unsupported_request_shapes as openai_unsupported_request_shapes,
)

_Raw = Mapping[str, object]


def unsupported_request_shapes(raw: _Raw) -> TranslationError | None:
    if carries_cache_control(raw.get("messages")) or carries_cache_control(
        raw.get("tools")
    ):
        return TranslationError.of_unsupported(
            "cache_control on a databricks message/tool: v1 MOVES message-level "
            "markers into a text block (and a whitespace-only message LOSES the "
            "marker with the sanitized block — a lossy interaction); the shared "
            "tool/message assembly strips the marker, so v1 serves its own move"
        )
    if _whitespace_only_content(raw.get("messages")):
        return TranslationError.of_unsupported(
            "whitespace-only/empty string message content: v1's "
            "_sanitize_empty_content REMOVES the content key (the message "
            "survives as a bare {role}); the IR keeps it as a text block"
        )
    return openai_unsupported_request_shapes(raw, name_fallback_user_only=True)


def _whitespace_only_content(messages: object) -> bool:
    if not isinstance(messages, Sequence) or isinstance(messages, str):
        return False
    for item in cast(Sequence[object], messages):
        if not isinstance(item, Mapping):
            continue
        content = cast(_Raw, item).get("content")
        if isinstance(content, str) and content.strip() == "":
            return True
    return False
