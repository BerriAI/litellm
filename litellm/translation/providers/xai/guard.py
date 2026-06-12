"""Raw-shape fidelity guard for the xai (Grok) serializer.

Three xai-only checks run before the shared openai guard:

- ``web_search_options``: v1 reroutes xai chat to the Responses-API bridge
  BEFORE any chat seam (``responses_api_bridge_check``, main.py:982-984), so
  a chat-route v2 must never serve it.
- ``use_xai_oauth``: v1's ``validate_environment`` runs an interactive
  browser PKCE flow with a localhost callback server (llms/xai/oauth.py);
  envelope I/O the v2 surface cannot reproduce.
- tool definitions carrying a ``strict`` key anywhere BELOW the function
  level: v1's ``filter_value_from_dict(tool, "strict")`` deletes EVERY key
  named ``strict`` at any depth (including JSON-schema properties literally
  named ``strict``); v2 reproduces only the standard function-level strip.

The shared openai guard then runs with ``name_fallback_user_only``: v1's xai
transform strips message ``name`` from every non-user role
(``strip_name_from_messages``), so the IR's name-drop IS v1's behavior there
and only a user-message ``name`` (forwarded verbatim by v1) falls back.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

from ...errors import TranslationError
from ..openai_compat.guard import explicit_stream_false
from ..openai_compat.guard import (
    unsupported_request_shapes as openai_unsupported_request_shapes,
)

_Raw = Mapping[str, object]


def unsupported_request_shapes(raw: _Raw) -> TranslationError | None:
    if raw.get("web_search_options") is not None:
        return TranslationError.of_unsupported(
            "web_search_options on xai: v1 reroutes the request to the "
            "Responses-API bridge above the chat seam "
            "(responses_api_bridge_check); v1 owns it"
        )
    if raw.get("use_xai_oauth"):
        # DEFENSE-IN-DEPTH ONLY, unreachable through translation_seam's
        # _raw_openai_body today: use_xai_oauth is a litellm param
        # (all_litellm_params), not an OpenAI param, so completion() never
        # places it in non_default_params/optional_param_args. The REAL
        # protection is the seam obligation (CLAUDE.md): the xai fork must
        # either include the kwarg in the raw body it routes (making this
        # arm live) or fall back on it BEFORE building deps, and must pin
        # that with a completion()-level test before the flag turns on.
        # test_use_xai_oauth_guard_reachability pins the classification facts
        # this analysis rests on.
        return TranslationError.of_unsupported(
            "use_xai_oauth: v1 runs an interactive browser PKCE flow inside "
            "validate_environment (llms/xai/oauth.py); v1 owns it"
        )
    stream_false = explicit_stream_false(raw)
    if stream_false is not None:
        return stream_false
    reason = _nested_tool_strict_reason(raw)
    if reason is not None:
        return TranslationError.of_unsupported(reason)
    return openai_unsupported_request_shapes(raw, name_fallback_user_only=True)


def _nested_tool_strict_reason(raw: _Raw) -> str | None:
    raw_tools = raw.get("tools")
    if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, str):
        return None
    for tool in cast(Sequence[object], raw_tools):
        if not isinstance(tool, Mapping):
            continue
        entry = cast(_Raw, tool)
        function = entry.get("function")
        if not isinstance(function, Mapping):
            continue
        for key, value in cast(_Raw, function).items():
            if key == "strict":
                continue  # the standard slot; the serializer strips it like v1
            if _contains_strict_key(value, 0):
                return (
                    "tool definition carries a nested 'strict' key below the "
                    "function level; v1's filter_value_from_dict deletes it "
                    "at every depth"
                )
    return None


def _contains_strict_key(value: object, depth: int) -> bool:
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        # exhaustion never admits: fall back (the polarity integration
        # db99d00be5 set for cap exhaustion; strictly widens the fallback)
        return True
    if isinstance(value, Mapping):
        mapping = cast(_Raw, value)
        return any(
            key == "strict" or _contains_strict_key(item, depth + 1)
            for key, item in mapping.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, str):
        return any(
            _contains_strict_key(item, depth + 1)
            for item in cast(Sequence[object], value)
        )
    return False
