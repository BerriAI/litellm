"""
Shared helpers for guardrail hooks: extract text from a request body
regardless of whether it uses Chat Completions ``messages``, Responses-API
``input``, or multimodal list-format ``content`` parts.

Hooks that only check ``data["messages"]`` for string content silently
skip the other shapes — these helpers normalise that so every hook sees
every text fragment.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Final, TypeGuard

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
TEXT_CONTENT_CALL_TYPES: Final[frozenset[str]] = frozenset({"completion", "acompletion", "aresponses"})


def is_text_content_call_type(call_type: str) -> bool:
    """Return True if ``call_type`` carries free-form text that text
    guardrails should inspect (Chat Completions or Responses API)."""
    return call_type in TEXT_CONTENT_CALL_TYPES


# Call types whose request body carries no conversation at all. Embeddings carry
# ``input`` — documents being indexed, not a prompt — which
# :func:`build_inspection_messages` would lift into synthetic chat messages.
#
# Deny-list on purpose: ``TEXT_CONTENT_CALL_TYPES`` above omits conversational
# call types (``anthropic_messages``, ``responses``, ``call_mcp_tool``), so a
# blocking guardrail gated on that allow-list would stop inspecting real chat
# traffic. Testing this instead leaves an unrecognised call type inspected.
NON_CONVERSATIONAL_CALL_TYPES: Final[frozenset[str]] = frozenset({"embedding", "aembedding"})


def is_non_conversational_call_type(call_type: str) -> bool:
    """Return True if ``call_type``'s body carries no conversation to inspect."""
    return call_type in NON_CONVERSATIONAL_CALL_TYPES


TEXT_PART_TYPES: Final[frozenset[str]] = frozenset(
    {"text", "input_text", "output_text", "summary_text", "reasoning_text"}
)

# Responses-API item types whose ``output`` field carries user/tool text
# that guardrails should inspect.  ``function_call_output`` is the
# built-in shape; ``custom_tool_call_output`` is the custom-tool
# counterpart (see ``ChatCompletionCustomToolCallOutput``).
_OUTPUT_ITEM_TYPES: Final[frozenset[str]] = frozenset(
    {"function_call_output", "custom_tool_call_output", "tool_search_output"}
)
_TOOL_SEARCH_TOOL_TEXT_KEYS: Final[tuple[str, ...]] = ("description",)


def _is_object_list(value: object) -> TypeGuard[list[object]]:  # guard-ok: narrows raw JSON lists
    return isinstance(value, list)


def _is_object_dict(value: object) -> TypeGuard[dict[str, object]]:  # guard-ok: narrows raw JSON objects
    return isinstance(value, dict)


def _is_object_mapping(value: object) -> TypeGuard[Mapping[str, object]]:  # guard-ok: narrows raw JSON mappings
    return isinstance(value, Mapping)


def _part_text(part: Mapping[str, object]) -> str | None:
    """Return non-empty plaintext from any content part that carries ``text``."""
    if not isinstance(part, dict):
        return None
    text = part.get("text")
    if isinstance(text, str) and text:
        return text
    return None


def _iter_text_parts_in_content(content: object) -> Iterator[str]:
    """Yield text fragments from a ``message.content`` value (string or
    multimodal list). Non-text parts (images, audio, …) are skipped."""
    if isinstance(content, str):
        if content:
            yield content
    elif _is_object_list(content):
        for part in content:
            if isinstance(part, str):
                # A bare string in a content/input list is itself a text
                # fragment (Responses-API mixed-list shape).
                if part:
                    yield part
                continue
            if not isinstance(part, dict):
                continue
            text = _part_text(part)
            if text is not None:
                yield text


def _has_plain_text_content(content: object) -> bool:
    """Return True when in-place masking can replace content without losing parts."""
    if isinstance(content, str):
        return True
    if not _is_object_list(content):
        return False
    return all(isinstance(part, str) or (_is_object_mapping(part) and _part_text(part) is not None) for part in content)


def _tool_search_tool_text(tool: Mapping[str, object]) -> str:
    return "\n".join(
        str(value) for key in _TOOL_SEARCH_TOOL_TEXT_KEYS if isinstance(value := tool.get(key), str) and value
    )


def _coerce_input_to_messages(input_value: object) -> list[dict[str, object]]:
    """Coerce a Responses-API ``data["input"]`` value into chat-style messages."""
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    if not _is_object_list(input_value):
        return []
    messages: list[dict[str, object]] = []  # mutable-ok: accumulated inspection messages
    for item in input_value:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
        elif _is_object_dict(item):
            if _part_text(item) is not None:
                messages.append({"role": item.get("role") or "user", "content": [item]})
            elif item.get("type") == "reasoning":
                if "content" in item:
                    messages.append(
                        {  # mutable-ok: append reasoning content
                            "role": item.get("role") or "assistant",
                            "content": item["content"],
                        }
                    )
                if _is_object_list(item.get("summary")):
                    messages.append(
                        {  # mutable-ok: append reasoning summary
                            "role": item.get("role") or "assistant",
                            "content": item["summary"],
                        }
                    )
            elif "content" in item:
                messages.append({"role": item.get("role") or "user", "content": item["content"]})
            elif item.get("type") in _OUTPUT_ITEM_TYPES and "output" in item:
                messages.append({"role": item.get("role") or "tool", "content": item["output"]})
            elif item.get("type") == "tool_search_output" and _is_object_list(item.get("tools")):
                for tool_idx, tool in enumerate(item["tools"]):
                    if _is_object_mapping(tool):
                        messages.append(
                            {  # mutable-ok: inspection snapshot with a write-back index
                                "role": "tool",
                                "content": _tool_search_tool_text(tool),
                                "_tool_index": tool_idx,
                            }
                        )
    return messages


def _iter_inspection_messages(data: Mapping[str, object]) -> Iterator[object]:
    """Yield every message-like dict, walking ``messages`` AND ``input``."""
    messages: Final = data.get("messages")
    if _is_object_list(messages):
        yield from messages
    yield from _coerce_input_to_messages(data.get("input"))


def iter_message_text(data: Mapping[str, object]) -> Iterator[str]:
    """Yield every text fragment from ``messages`` AND ``input``.

    Walks every role (user, assistant, system, …) — guardrails inspect
    the entire conversation, not just user turns.
    """
    for message in _iter_inspection_messages(data):
        if not _is_object_dict(message):
            continue
        yield from _iter_text_parts_in_content(message.get("content"))


def walk_user_text(data: dict[str, Any], visit: Callable[[str], str]) -> int:
    """Rewrite every text fragment in place via ``visit``.

    Mutates ``data["messages"]`` and ``data["input"]``. Returns the number
    of fragments visited so callers can short-circuit when nothing was
    inspected.
    """
    visited = 0

    def _rewrite_content(content: object) -> object:
        nonlocal visited
        if isinstance(content, str):
            if content:
                visited += 1
                return visit(content)
            return content
        if _is_object_list(content):
            new_parts: Final[list[object]] = []
            for part in content:
                if isinstance(part, str) and part:
                    visited += 1
                    new_parts.append(visit(part))
                elif _is_object_dict(part) and _part_text(part) is not None:
                    visited += 1
                    new_parts.append({**part, "text": visit(part["text"])})
                else:
                    new_parts.append(part)
            return new_parts
        return content

    def _rewrite_tool_search_tool(tool: Mapping[str, object]) -> dict[str, object]:
        rewritten: Final = dict(tool)  # mutable-ok: fresh copy so guardrail rewrites do not mutate the original
        for key in _TOOL_SEARCH_TOOL_TEXT_KEYS:
            value = rewritten.get(key)
            if isinstance(value, str) and value:
                nonlocal visited
                visited += 1
                rewritten[key] = visit(value)
        return rewritten

    messages: Final = data.get("messages")
    if _is_object_list(messages):
        for message in messages:
            if _is_object_dict(message) and "content" in message:
                message["content"] = _rewrite_content(message["content"])

    input_value: Final = data.get("input")
    if isinstance(input_value, str):
        if input_value:
            visited += 1
            data["input"] = visit(input_value)
        return visited
    if _is_object_list(input_value):
        for idx, item in enumerate(input_value):
            if isinstance(item, str):
                if item:
                    visited += 1
                    input_value[idx] = visit(item)
            elif _is_object_dict(item):
                if _part_text(item) is not None:
                    visited += 1
                    input_value[idx] = {**item, "text": visit(item["text"])}  # mutable-ok: rewrite text part in place
                elif item.get("type") == "reasoning":
                    if "content" in item:
                        item["content"] = _rewrite_content(item["content"])
                    if _is_object_list(item.get("summary")):
                        item["summary"] = _rewrite_content(item["summary"])
                elif "content" in item:
                    item["content"] = _rewrite_content(item["content"])
                elif item.get("type") in _OUTPUT_ITEM_TYPES and "output" in item:
                    item["output"] = _rewrite_content(item["output"])
                elif item.get("type") == "tool_search_output" and _is_object_list(item.get("tools")):
                    tools: list[object] = item["tools"]
                    item["tools"] = [  # mutable-ok: rewrites forwarded tool-search results in place
                        _rewrite_tool_search_tool(tool) if _is_object_mapping(tool) else tool for tool in tools
                    ]
        return visited

    return visited


def is_string_batch_input(data: Mapping[str, object]) -> bool:
    """Return True when the only inspected content is an ``input`` list of plain
    strings, the /embeddings batch shape, which :func:`apply_redacted_messages_back`
    rewrites element-wise."""
    if "messages" in data:
        return False
    input_value: Final = data.get("input")
    return _is_object_list(input_value) and bool(input_value) and all(isinstance(item, str) for item in input_value)


def _apply_redacted_input_texts(input_value: list[object], response_texts: tuple[str, ...]) -> None:
    redacted_iter: Final[Iterator[str]] = iter(response_texts)
    for item_idx, item in enumerate(input_value):
        if isinstance(item, str):
            if item:
                input_value[item_idx] = next(redacted_iter)
        elif _is_object_dict(item):
            _apply_redacted_input_item(item, redacted_iter)


def _apply_redacted_input_item(item: dict[str, object], redacted_iter: Iterator[str]) -> None:
    if _part_text(item) is not None:
        item["text"] = next(redacted_iter)
    elif item.get("type") == "reasoning":
        if "content" in item and any(_iter_text_parts_in_content(item["content"])):
            item["content"] = next(redacted_iter)
        if _is_object_list(item.get("summary")) and any(_iter_text_parts_in_content(item["summary"])):
            item["summary"] = next(redacted_iter)
    elif "content" in item and any(_iter_text_parts_in_content(item["content"])):
        item["content"] = next(redacted_iter)
    elif item.get("type") in _OUTPUT_ITEM_TYPES and "output" in item:
        if any(_iter_text_parts_in_content(item["output"])):
            item["output"] = next(redacted_iter)
    elif item.get("type") == "tool_search_output" and _is_object_list(item.get("tools")):
        _apply_redacted_tool_search_tools(item, redacted_iter)


def _apply_redacted_tool_search_tools(item: dict[str, object], redacted_iter: Iterator[str]) -> None:
    rewritten_tools: list[object] = []  # mutable-ok: request tools must remain a JSON list
    tools: Final[object] = item["tools"]
    if not _is_object_list(tools):
        return
    for tool in tools:
        if _is_object_mapping(tool) and _tool_search_tool_text(tool):
            rewritten_tools.append(
                {  # mutable-ok: request tools must remain plain JSON dicts
                    **tool,
                    "description": next(redacted_iter),
                }
            )
        else:
            rewritten_tools.append(tool)
    item["tools"] = rewritten_tools


def _apply_redacted_string_input(data: dict[str, Any], redacted_text: str) -> None:
    data["input"] = redacted_text


def _apply_redacted_string_batch_input(data: dict[str, Any], redacted_messages: Sequence[object]) -> bool:
    batch: Final = data["input"]
    inspected_indices: Final = tuple(idx for idx, item in enumerate(batch) if item)
    if len(redacted_messages) != len(inspected_indices):
        return False
    if any(not isinstance(message, Mapping) or message.get("content") is None for message in redacted_messages):
        return False
    redacted_texts: Final = tuple(
        "\n".join(_iter_text_parts_in_content(message["content"])) for message in redacted_messages
    )
    for idx, text in zip(inspected_indices, redacted_texts):
        batch[idx] = text
    return True


def apply_redacted_messages_back(data: dict[str, Any], redacted_messages: Sequence[object]) -> bool:
    """Write redacted messages back to whichever field(s) the caller used.

    Mask/anonymize paths take a synthesised messages list (from
    :func:`build_inspection_messages`), get a redacted version back from a
    third-party guardrail, and need to rewrite the request body. Writing
    only to ``data["messages"]`` leaves the Responses-API ``data["input"]``
    field untouched, so the unredacted text still reaches the LLM.

    This helper updates both fields when both are present. A string batch
    (``/embeddings`` ``input`` list) is rewritten element-wise: the n-th
    redacted message replaces the n-th non-empty element, because
    :func:`build_inspection_messages` emits one message per non-empty string.
    When both ``messages`` and structured ``input`` are present, their
    redactions are grouped by source field. A single redaction can update both
    fields only when it is the sole response and their inspected inputs match
    one-to-one.
    Responses-API ``tool_search_output`` tool descriptions are rewritten in
    place, preserving the surrounding item and non-text fields. The
    ``tool_search_output`` fallback ``output`` field is rewritten the same way.

    Returns False, leaving ``data`` untouched, when a batch response does not
    carry exactly one message per inspected element: a partial rewrite would
    forward the remaining originals unredacted. Callers must block on False.
    """
    if is_string_batch_input(data):
        return _apply_redacted_string_batch_input(data, redacted_messages)
    input_value: Final = data.get("input")

    if "messages" not in data:
        return _apply_redacted_input_without_messages(data, input_value, redacted_messages)

    message_count: Final[int] = _inspection_count("messages", data["messages"])
    input_count: Final[int] = _inspection_count("input", input_value) if isinstance(input_value, (str, list)) else 0
    if message_count > len(redacted_messages):
        return False
    if len(redacted_messages) == message_count:
        original_messages: Final = build_inspection_messages(data)
        if (
            message_count == 1
            and input_count == 1
            and original_messages[0]["content"] == original_messages[1]["content"]
        ):
            matched_redacted_input_text: Final = _text_of_message(redacted_messages[0])
            if isinstance(matched_redacted_input_text, str):
                if isinstance(input_value, str):
                    _apply_redacted_string_input(data, matched_redacted_input_text)
                else:
                    _structured_redactions_apply(
                        input_value,
                        ({"content": matched_redacted_input_text},),  # mutable-ok: one-shot redaction message
                    )
            else:
                return False
        elif input_count:
            return False
        data["messages"] = redacted_messages
        return True
    input_redactions: Final = redacted_messages[message_count:]
    if input_count != len(input_redactions):
        return False
    if isinstance(input_value, str):
        redacted_input_text: Final = _text_of_message(input_redactions[0])
        if not isinstance(redacted_input_text, str):
            return False
        _apply_redacted_string_input(data, redacted_input_text)
    else:
        if not _is_object_list(input_value) or not _structured_redactions_apply(input_value, input_redactions):
            return False
    data["messages"] = redacted_messages[:message_count]
    return True


def _inspection_count(field: str, value: object) -> int:
    return len(build_inspection_messages({field: value}))  # mutable-ok: one-shot field wrapper


def _apply_redacted_input_without_messages(
    data: dict[str, Any], input_value: object, redacted_messages: Sequence[object]
) -> bool:
    if isinstance(input_value, str):
        if len(redacted_messages) != 1:
            return False
        input_text: Final = _text_of_message(redacted_messages[0])
        if not isinstance(input_text, str):
            return False
        _apply_redacted_string_input(data, input_text)
        return True
    if _is_object_list(input_value):
        return _structured_redactions_apply(input_value, redacted_messages)
    return True


def _redacted_texts(messages: Sequence[object]) -> tuple[str, ...]:
    return tuple(text for msg in messages if isinstance(text := _text_of_message(msg), str))


def _structured_redactions_apply(input_value: list[object], redactions: Sequence[object]) -> bool:
    input_texts: Final = _redacted_texts(redactions)
    if len(input_texts) != len(redactions):
        return False
    if len(input_texts) != _inspection_count("input", input_value):
        return False
    _apply_redacted_input_texts(input_value, input_texts)
    return True


def _text_of_message(message: object) -> str | None:
    if not isinstance(message, Mapping):
        return None
    content: Final = message.get("content")
    if isinstance(content, str):
        return content
    if _is_object_list(content):
        return "\n".join(_iter_text_parts_in_content(content))
    return None


def _has_non_string_input_item(item: object) -> bool:
    if not _is_object_dict(item):
        return True
    if _part_text(item) is not None:
        return False
    if item.get("type") == "reasoning":
        return ("content" in item and not _has_plain_text_content(item["content"])) or (
            "summary" in item and not _has_plain_text_content(item["summary"])
        )
    if "content" in item:
        return not _has_plain_text_content(item["content"])
    if item.get("type") in _OUTPUT_ITEM_TYPES and "output" in item:
        return not _has_plain_text_content(item["output"])
    return False


def has_non_string_content(data: Mapping[str, object]) -> bool:
    """Return True if any inspected content is not a plain string.

    Used by hooks whose mask/redact path operates on string offsets and
    therefore cannot preserve multimodal non-text parts. Such hooks should
    degrade to block-on-detect when this returns True so image/audio parts
    are not silently stripped during in-place masking.
    """
    messages: Final = data.get("messages")
    if _is_object_list(messages):
        for message in messages:
            if _is_object_dict(message) and not isinstance(message.get("content"), str):
                if message.get("content") is not None:
                    return True
    input_value: Final = data.get("input")
    if input_value is None or isinstance(input_value, str):
        return False
    if not _is_object_list(input_value):
        return True
    return any(_has_non_string_input_item(item) for item in input_value)


def build_inspection_messages(data: dict[str, Any]) -> list[dict[str, str]]:
    """Synthesize a chat-style messages list for posting to a guardrail API.

    Each returned message has a plain-string ``content`` — multimodal text
    parts are joined with newlines and Responses-API ``input`` is lifted
    into synthetic messages. Messages with no inspectable text are dropped.

    Hooks that POST ``{"messages": [...]}`` to an external service should
    call this instead of ``data.get("messages", [])`` so the Responses API
    and multimodal content are covered.
    """
    flattened: Final[list[dict[str, str]]] = []
    for message in _iter_inspection_messages(data):
        if not _is_object_dict(message):
            continue
        if "_tool_index" in message and isinstance(message.get("content"), str):
            flattened.append(
                {  # mutable-ok: fresh synthetic message, not stored in the original request
                    "role": message.get("role") or "tool",
                    "content": message["content"],
                }
            )
            continue
        text = "\n".join(_iter_text_parts_in_content(message.get("content")))
        if not text:
            continue
        role = message.get("role", "user") or "user"
        flattened.append({"role": role, "content": text})
    return flattened
