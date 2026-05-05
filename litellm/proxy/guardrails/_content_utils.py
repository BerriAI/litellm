"""
Shared helpers for guardrail hooks: extract text from a request body
regardless of whether it uses Chat Completions ``messages``, Responses-API
``input``, or multimodal list-format ``content`` parts.

Hooks that only check ``data["messages"]`` for string content silently
skip the other shapes — these helpers normalise that so every hook sees
every text fragment.
"""

from typing import Any, Callable, Dict, FrozenSet, Iterator, List


# Call types whose body carries free-form chat / prompt text that
# text-content guardrails (banned keywords, content moderation, secret
# detection, …) should inspect. The proxy ingress passes ``route_type``
# straight through as ``call_type``, so the literal values here are
# what the guardrail dispatcher actually receives:
#
#   /v1/chat/completions   -> "acompletion"
#   /v1/responses          -> "aresponses"
#
# ``"completion"`` is included for SDK / internal callers that invoke
# ``pre_call_hook`` directly with the sync name. Embedding, moderation,
# audio, and transcription endpoints are deliberately excluded — text
# guardrails on those paths are a separate scope.
TEXT_CONTENT_CALL_TYPES: FrozenSet[str] = frozenset(
    {"completion", "acompletion", "aresponses"}
)


def is_text_content_call_type(call_type: str) -> bool:
    """Return True if ``call_type`` carries free-form text that text
    guardrails should inspect (Chat Completions or Responses API)."""
    return call_type in TEXT_CONTENT_CALL_TYPES


def _iter_text_parts_in_content(content: Any) -> Iterator[str]:
    """Yield text fragments from a ``message.content`` value (string or
    multimodal list). Non-text parts (images, audio, …) are skipped."""
    if isinstance(content, str):
        if content:
            yield content
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                # A bare string in a content/input list is itself a text
                # fragment (Responses-API mixed-list shape).
                if part:
                    yield part
                continue
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str) and text:
                    yield text


def _coerce_input_to_messages(input_value: Any) -> List[Dict[str, Any]]:
    """Coerce a Responses-API ``data["input"]`` value into chat-style messages."""
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    if isinstance(input_value, list):
        if input_value and all(
            isinstance(item, dict) and "role" in item for item in input_value
        ):
            return list(input_value)
        # Mixed lists (content-part dicts + bare strings) and pure
        # string/dict lists all become a single user message; the content
        # iterator below handles each element type uniformly.
        return [{"role": "user", "content": input_value}]
    return []


def _iter_inspection_messages(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield every message-like dict, walking ``messages`` AND ``input``."""
    messages = data.get("messages")
    if isinstance(messages, list):
        yield from messages
    yield from _coerce_input_to_messages(data.get("input"))


def iter_message_text(data: Dict[str, Any]) -> Iterator[str]:
    """Yield every text fragment from ``messages`` AND ``input``.

    Walks every role (user, assistant, system, …) — guardrails inspect
    the entire conversation, not just user turns.
    """
    for message in _iter_inspection_messages(data):
        if not isinstance(message, dict):
            continue
        yield from _iter_text_parts_in_content(message.get("content"))


def walk_user_text(data: Dict[str, Any], visit: Callable[[str], str]) -> int:
    """Rewrite every text fragment in place via ``visit``.

    Mutates ``data["messages"]`` and ``data["input"]``. Returns the number
    of fragments visited so callers can short-circuit when nothing was
    inspected.
    """
    visited = 0

    def _rewrite_content(content: Any) -> Any:
        nonlocal visited
        if isinstance(content, str):
            if content:
                visited += 1
                return visit(content)
            return content
        if isinstance(content, list):
            new_parts: List[Any] = []
            for part in content:
                if isinstance(part, str) and part:
                    visited += 1
                    new_parts.append(visit(part))
                elif (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                    and part["text"]
                ):
                    visited += 1
                    new_parts.append({**part, "text": visit(part["text"])})
                else:
                    new_parts.append(part)
            return new_parts
        return content

    messages = data.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                message["content"] = _rewrite_content(message["content"])

    input_value = data.get("input")
    if isinstance(input_value, str):
        if input_value:
            visited += 1
            data["input"] = visit(input_value)
        return visited
    if isinstance(input_value, list):
        # List of full messages: rewrite each message's content.
        if input_value and all(
            isinstance(item, dict) and "role" in item for item in input_value
        ):
            for item in input_value:
                if "content" in item:
                    item["content"] = _rewrite_content(item["content"])
            return visited
        # List of content parts and/or bare strings: rewrite in place.
        for idx, item in enumerate(input_value):
            if isinstance(item, str) and item:
                visited += 1
                input_value[idx] = visit(item)
            elif (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
                and item["text"]
            ):
                visited += 1
                input_value[idx] = {**item, "text": visit(item["text"])}
        return visited

    return visited


def apply_redacted_messages_back(
    data: Dict[str, Any], redacted_messages: List[Dict[str, Any]]
) -> None:
    """Write redacted messages back to whichever field(s) the caller used.

    Mask/anonymize paths take a synthesised messages list (from
    :func:`build_inspection_messages`), get a redacted version back from a
    third-party guardrail, and need to rewrite the request body. Writing
    only to ``data["messages"]`` leaves the Responses-API ``data["input"]``
    field untouched, so the unredacted text still reaches the LLM.

    This helper updates both fields when both are present.
    """
    if "messages" in data:
        data["messages"] = redacted_messages
    if isinstance(data.get("input"), str):
        text_parts: List[str] = []
        for msg in redacted_messages:
            if not isinstance(msg, dict):
                continue
            text_parts.extend(_iter_text_parts_in_content(msg.get("content")))
        data["input"] = "\n".join(text_parts)


def has_non_string_content(data: Dict[str, Any]) -> bool:
    """Return True if any inspected content is not a plain string.

    Used by hooks whose mask/redact path operates on string offsets and
    therefore cannot preserve multimodal non-text parts. Such hooks should
    degrade to block-on-detect when this returns True so image/audio parts
    are not silently stripped during in-place masking.
    """
    messages = data.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and not isinstance(
                message.get("content"), str
            ):
                if message.get("content") is not None:
                    return True
    input_value = data.get("input")
    if input_value is not None and not isinstance(input_value, str):
        return True
    return False


def build_inspection_messages(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Synthesize a chat-style messages list for posting to a guardrail API.

    Each returned message has a plain-string ``content`` — multimodal text
    parts are joined with newlines and Responses-API ``input`` is lifted
    into synthetic messages. Messages with no inspectable text are dropped.

    Hooks that POST ``{"messages": [...]}`` to an external service should
    call this instead of ``data.get("messages", [])`` so the Responses API
    and multimodal content are covered.
    """
    flattened: List[Dict[str, str]] = []
    for message in _iter_inspection_messages(data):
        if not isinstance(message, dict):
            continue
        text = "\n".join(_iter_text_parts_in_content(message.get("content")))
        if not text:
            continue
        role = message.get("role", "user") or "user"
        flattened.append({"role": role, "content": text})
    return flattened
