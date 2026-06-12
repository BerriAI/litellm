"""Raw-shape fidelity guard for the OpenAI passthrough serializer.

v1's OpenAI request transform is a near-passthrough (five touches), so byte
parity through the IR is only possible for shapes the shared inbound parse
round-trips losslessly. The parse normalizes wire forms v1 forwards verbatim
(string-vs-list content, message ``name``, image ``detail``, the
max-tokens-key split, empty-list-vs-absent), so
requests carrying them return a typed error here and the seam serves them
through v1 unchanged. Runs over the UNTRUSTED raw body BEFORE parse; every
check is structural and conservative: a guard can only widen the fallback
surface, never change a served body.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from ...errors import TranslationError

_Raw = Mapping[str, object]


def unsupported_request_shapes(
    raw: _Raw, *, name_fallback_user_only: bool = False
) -> TranslationError | None:
    """``name_fallback_user_only``: providers whose v1 transform STRIPS the
    message ``name`` from non-user roles (xai ``strip_name_from_messages``)
    only need the fallback for user messages, where v1 forwards it verbatim;
    the IR's name-drop IS v1's behavior on the other roles."""
    reason = (
        _params_reason(raw)
        or _tools_reason(raw)
        or _messages_reason(raw, name_fallback_user_only)
    )
    if reason is None:
        return None
    return TranslationError.of_unsupported(f"{reason}; v1 forwards the original shape")


def _as_map(value: object) -> _Raw | None:
    if isinstance(value, Mapping):
        return cast(_Raw, value)
    return None


def _as_seq(value: object) -> Sequence[object] | None:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return cast(Sequence[object], value)
    return None


def _params_reason(raw: _Raw) -> str | None:
    if (
        raw.get("max_tokens") is not None
        and raw.get("max_completion_tokens") is not None
    ):
        return "both max_tokens and max_completion_tokens sent (the IR keeps one)"
    if isinstance(raw.get("stop"), str):
        return "string-form stop (the IR keeps a list)"
    if raw.get("stop") == []:
        return "empty stop list (absent-vs-[] is lost in the IR)"
    choice = _as_map(raw.get("tool_choice"))
    if choice is not None and (
        choice.get("type") != "function" or "function" not in choice
    ):
        return "dict-form tool_choice without type=function (the IR keeps the tag)"
    return None


def _tools_reason(raw: _Raw) -> str | None:
    tools = _as_seq(raw.get("tools"))
    if tools is None:
        return None
    if len(tools) == 0:
        return "empty tools list (absent-vs-[] is lost in the IR)"
    for tool in tools:
        entry = _as_map(tool)
        if entry is not None and entry.get("type") == "custom":
            return "custom-type tool (the IR keeps function tools only)"
    return None


def _messages_reason(raw: _Raw, name_fallback_user_only: bool = False) -> str | None:
    messages = _as_seq(raw.get("messages"))
    if messages is None:
        return None
    previous_role: object = None
    seen_non_system = False
    for item in messages:
        entry = _as_map(item)
        if entry is None:
            # parse rejects the non-object message with a boundary error;
            # keep scanning so the guard stays locally conservative too
            continue
        role = entry.get("role")
        if entry.get("name") is not None and (
            not name_fallback_user_only or role == "user"
        ):
            return "message name field (not carried by the IR)"
        reason = _message_reason(entry, role, seen_non_system)
        if reason is not None:
            return reason
        if role in ("user", "assistant") and role == previous_role:
            return f"consecutive {role} messages (the IR merges adjacent turns)"
        previous_role = role
        seen_non_system = seen_non_system or role != "system"
    return None


def _message_reason(entry: _Raw, role: object, seen_non_system: bool) -> str | None:
    if role == "system":
        return _system_reason(entry, seen_non_system)
    if role == "user":
        return _user_reason(entry)
    if role == "assistant":
        return _assistant_reason(entry)
    if role == "tool":
        return _tool_reason(entry)
    return None


def _system_reason(entry: _Raw, seen_non_system: bool) -> str | None:
    if seen_non_system:
        return "system message after the first turn (the IR hoists system text)"
    content = entry.get("content")
    if content is None or content == "":
        return "empty system message (dropped by the IR)"
    if _as_seq(content) is not None:
        return "list-form system content (the IR keeps text only)"
    return None


def _user_reason(entry: _Raw) -> str | None:
    content = entry.get("content")
    if content is None:
        return "user message without content (dropped by the IR)"
    if isinstance(content, str):
        return None
    items = _as_seq(content)
    if items is None:
        return None
    if len(items) == 0:
        return "empty user content list (dropped by the IR)"
    parts = [part for item in items if (part := _as_map(item)) is not None]
    if len(parts) == 1 and parts[0].get("type") == "text":
        return "single-text content list (string-vs-list form is lost in the IR)"
    for part in parts:
        image_url = _as_map(part.get("image_url"))
        if image_url is not None and (
            image_url.get("detail") is not None or image_url.get("format") is not None
        ):
            return "image_url detail/format keys (not carried by the IR)"
    return None


def _assistant_reason(entry: _Raw) -> str | None:
    for field in ("thinking_blocks", "refusal", "annotations", "audio"):
        if entry.get(field) is not None:
            return f"assistant {field} (no lossless OpenAI re-encode from the IR)"
    tool_calls = entry.get("tool_calls")
    if "tool_calls" in entry and tool_calls is None:
        return "explicit null tool_calls (key presence is lost in the IR)"
    content = entry.get("content")
    if content is None and tool_calls is None:
        return "assistant message without content (dropped by the IR)"
    if tool_calls is not None and "content" not in entry:
        return (
            "tool-call message without a content key (key presence is lost in the IR)"
        )
    items = _as_seq(content)
    if items is not None:
        reason = _assistant_content_reason(items)
        if reason is not None:
            return reason
    return _tool_calls_reason(tool_calls)


def _assistant_content_reason(content: Sequence[object]) -> str | None:
    if len(content) == 0:
        return "empty assistant content list (dropped by the IR)"
    parts = [part for item in content if (part := _as_map(item)) is not None]
    if len(parts) == 1 and parts[0].get("type") == "text":
        return "single-text content list (string-vs-list form is lost in the IR)"
    for part in parts:
        if part.get("type") != "text":
            return "non-text assistant content part (no lossless OpenAI re-encode)"
    return None


def _tool_calls_reason(tool_calls: object) -> str | None:
    calls = _as_seq(tool_calls)
    if calls is None:
        return None
    for item in calls:
        call = _as_map(item)
        if call is None:
            # parse rejects the non-object tool_call; keep scanning
            continue
        if "index" in call:
            return "tool_call index key (not carried by the IR)"
        if call.get("type") != "function":
            return "non-function tool_call (dropped by the IR, forwarded by v1)"
        function = _as_map(call.get("function"))
        arguments = function.get("arguments") if function is not None else None
        if not isinstance(arguments, str):
            return "tool_call without string arguments (the IR re-dumps a {} default)"
        # string arguments need no spacing check: the IR carries the verbatim
        # wire bytes (ToolUse.arguments_raw) and the serializer re-emits them
    return None


def _tool_reason(entry: _Raw) -> str | None:
    content = entry.get("content")
    if content is None:
        return "tool message without content (the IR rewrites it to an empty string)"
    if _as_seq(content) is not None:
        return "list-form tool content (string-vs-list form is lost in the IR)"
    return None
