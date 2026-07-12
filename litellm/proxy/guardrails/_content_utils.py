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
TEXT_CONTENT_CALL_TYPES: FrozenSet[str] = frozenset({"completion", "acompletion", "aresponses"})


def is_text_content_call_type(call_type: str) -> bool:
    """Return True if ``call_type`` carries free-form text that text
    guardrails should inspect (Chat Completions or Responses API)."""
    return call_type in TEXT_CONTENT_CALL_TYPES


TEXT_PART_TYPES: Frozenset[str] = frozenset({"text", "input_text", "output_text", "summary_text"})


def _iter_summary_texts(summary: Any) -> Iterator[str]:
    """Yield text fragments from a ``reasoning`` item's ``summary`` list —
    ``summary_text`` parts and bare strings; anything else is skipped."""
    if not isinstance(summary, list):
        return
    for summary_part in summary:
        if isinstance(summary_part, str):
            if summary_part:
                yield summary_part
            continue
        if not isinstance(summary_part, dict):
            continue
        if summary_part.get("type") in TEXT_PART_TYPES:
            summary_text = summary_part.get("text")
            if isinstance(summary_text, str) and summary_text:
                yield summary_text

# Responses-API item types whose ``output`` field carries user/tool text
# that guardrails should inspect.  ``function_call_output`` is the
# built-in shape; ``custom_tool_call_output`` is the custom-tool
# counterpart (see ``ChatCompletionCustomToolCallOutput``).
_OUTPUT_ITEM_TYPES: frozenset[str] = frozenset({"function_call_output", "custom_tool_call_output"})


def _iter_text_parts_in_content(content: Any) -> Iterator[str]:
    """Yield text fragments from a ``message.content`` value (string or
    multimodal list). Non-text parts (images, audio, …) are skipped.

    Also descends into ``reasoning`` items whose ``summary`` list may
    contain ``summary_text`` parts carrying chain-of-thought text that
    guardrails need to inspect/redact.
    """
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
            if part.get("type") in TEXT_PART_TYPES:
                text = part.get("text")
                if isinstance(text, str) and text:
                    yield text
            elif part.get("type") == "reasoning":
                # Reasoning items carry a ``summary`` list of
                # ``{"type": "summary_text", "text": "..."}`` parts.
                yield from _iter_summary_texts(part.get("summary"))


def _coerce_input_to_messages(input_value: Any) -> List[Dict[str, Any]]:
    """Coerce a Responses-API ``data["input"]`` value into chat-style messages."""
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    if not isinstance(input_value, list):
        return []
    messages: List[Dict[str, Any]] = []
    for item in input_value:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
        elif isinstance(item, dict):
            if item.get("type") in TEXT_PART_TYPES:
                messages.append({"role": item.get("role") or "user", "content": [item]})
            elif "content" in item:
                messages.append({"role": item.get("role") or "user", "content": item["content"]})
            elif item.get("type") in _OUTPUT_ITEM_TYPES and "output" in item:
                messages.append({"role": item.get("role") or "tool", "content": item["output"]})
            elif item.get("type") == "reasoning":
                # Reasoning items carry chain-of-thought text in their
                # ``summary`` list.  Synthesise a message so guardrails
                # can inspect/redact that text.
                summary = item.get("summary")
                if isinstance(summary, list):
                    messages.append({"role": "assistant", "content": summary})
    return messages


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

    def _rewrite_summary(summary: list[Any]) -> list[Any]:
        nonlocal visited
        new_parts: List[Any] = []
        for summary_part in summary:
            if isinstance(summary_part, str) and summary_part:
                visited += 1
                new_parts.append(visit(summary_part))
            elif (
                isinstance(summary_part, dict)
                and summary_part.get("type") in TEXT_PART_TYPES
                and isinstance(summary_part.get("text"), str)
                and summary_part["text"]
            ):
                visited += 1
                new_parts.append({**summary_part, "text": visit(summary_part["text"])})
            else:
                new_parts.append(summary_part)
        return new_parts

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
                    and part.get("type") in TEXT_PART_TYPES
                    and isinstance(part.get("text"), str)
                    and part["text"]
                ):
                    visited += 1
                    new_parts.append({**part, "text": visit(part["text"])})
                elif (
                    isinstance(part, dict) and part.get("type") == "reasoning" and isinstance(part.get("summary"), list)
                ):
                    new_parts.append({**part, "summary": _rewrite_summary(part["summary"])})
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
        for idx, item in enumerate(input_value):
            if isinstance(item, str):
                if item:
                    visited += 1
                    input_value[idx] = visit(item)
            elif isinstance(item, dict):
                if item.get("type") in TEXT_PART_TYPES:
                    if isinstance(item.get("text"), str) and item["text"]:
                        visited += 1
                        input_value[idx] = {**item, "text": visit(item["text"])}
                elif "content" in item:
                    item["content"] = _rewrite_content(item["content"])
                elif item.get("type") in _OUTPUT_ITEM_TYPES and "output" in item:
                    item["output"] = _rewrite_content(item["output"])
                elif item.get("type") == "reasoning":
                    summary = item.get("summary")
                    if isinstance(summary, list):
                        item["summary"] = _rewrite_content(summary)
        return visited

    return visited


def apply_redacted_messages_back(data: Dict[str, Any], redacted_messages: List[Dict[str, Any]]) -> None:
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
            if isinstance(message, dict) and not isinstance(message.get("content"), str):
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
