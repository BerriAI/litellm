"""
Shared helpers for guardrail hooks: extract text from a request body
regardless of whether it uses Chat Completions ``messages``, Responses-API
``input``, or multimodal list-format ``content`` parts.

Hooks that only check ``data["messages"]`` for string content silently
skip the other shapes — these helpers normalise that so every hook sees
every text fragment.
"""

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import count
from typing import Any, Final

from pydantic import TypeAdapter

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
_OUTPUT_ITEM_TYPES: Final[frozenset[str]] = frozenset({"function_call_output", "custom_tool_call_output"})


def _part_text(part: Mapping[str, object]) -> str | None:
    """Return non-empty plaintext from any content part that carries ``text``."""
    if not isinstance(part, dict):
        return None
    text = part.get("text")
    if isinstance(text, str) and text:
        return text
    return None


_OBJECT_LIST: Final[TypeAdapter[Sequence[object]]] = TypeAdapter(Sequence[object])
_TEXT_OBJECT: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


def _object_list(value: object) -> Sequence[object] | None:
    return _OBJECT_LIST.validate_python(value) if isinstance(value, list) else None


def _text_object(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, dict):
        return None
    try:
        return _TEXT_OBJECT.validate_python(value)
    except ValueError:
        return None


def _is_json_part(part: Mapping[str, object]) -> bool:
    return part.get("type") == "input_json" and "data" in part


def iter_json_fields(value: object) -> Iterator[tuple[str | None, str]]:
    pending: Final[list[tuple[object, str | None]]] = [(value, None)]  # mutable-ok: owned DFS stack avoids recursion
    while pending:
        current, key = pending.pop()
        if isinstance(current, str):
            if current:
                yield key, current
            continue
        fields = _text_object(current)
        if fields is not None:
            pending.extend((child, child_key) for child_key, child in reversed(tuple(fields.items())))
            continue
        pending.extend((child, key) for child in reversed(_object_list(current) or ()))


def iter_json_keys(value: object) -> Iterator[str]:
    pending: Final[list[object]] = [value]  # mutable-ok: owned DFS stack avoids recursion
    while pending:
        current = pending.pop()
        fields = _text_object(current)
        if fields is not None:
            yield from (key for key in fields if key)
            pending.extend(reversed(tuple(fields.values())))
            continue
        pending.extend(reversed(_object_list(current) or ()))


def _json_has_text(value: object, *, with_keys: bool) -> bool:
    has_value: Final = next(iter_json_fields(value), None) is not None
    return has_value or (with_keys and next(iter_json_keys(value), None) is not None)


def distinct_keys(renames: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    return tuple(_claim_names(renames))


def _claim_names(renames: Sequence[tuple[str, str]]) -> Iterator[str]:
    taken: Final[set[str]] = {new for old, new in renames if new == old}  # mutable-ok: names already used in one object
    candidates: Final[dict[str, Iterator[str]]] = {}  # mutable-ok: resumes each name's suffix search where it stopped
    for old, new in renames:
        if new == old:
            yield old
            continue
        name = next(candidate for candidate in candidates.setdefault(new, _numbered(new)) if candidate not in taken)
        taken.add(name)
        yield name


def _numbered(name: str) -> Iterator[str]:
    yield name
    yield from (f"{name} ({number})" for number in count(2))


def _renamed_fields(fields: Mapping[str, object], visit: Callable[[str], str]) -> tuple[tuple[str, object], ...]:
    names: Final = distinct_keys(tuple((key, visit(key) if key else key) for key in fields))
    return tuple(zip(names, fields.values()))


@dataclass(frozen=True, slots=True)
class _JsonFrame:
    source: object
    write: Callable[[object], None]


def map_json_text(value: object, visit: Callable[[str], str], *, rewrite_keys: bool = False) -> object:
    root: Final[list[object]] = [None]  # mutable-ok: one owned root slot receives the iterative result
    pending: Final[list[_JsonFrame]] = [_JsonFrame(value, partial(root.__setitem__, 0))]  # mutable-ok: owned DFS stack
    while pending:
        frame = pending.pop()
        current = frame.source
        if isinstance(current, str):
            frame.write(visit(current) if current else current)
            continue
        fields = _text_object(current)
        if fields is not None:
            renamed = _renamed_fields(fields, visit) if rewrite_keys else tuple(fields.items())
            mapped: dict[str, object] = {key: None for key, _ in renamed}  # mutable-ok: owned output slots
            frame.write(mapped)
            pending.extend(_JsonFrame(child, partial(mapped.__setitem__, key)) for key, child in reversed(renamed))
            continue
        children = _object_list(current)
        if children is None:
            frame.write(current)
            continue
        items: list[object] = [None] * len(children)  # mutable-ok: fresh indexed output slots
        frame.write(items)
        pending.extend(
            _JsonFrame(child, partial(items.__setitem__, index))
            for index, child in reversed(tuple(enumerate(children)))
        )
    return root[0]


def _iter_text_parts_in_content(content: object) -> Iterator[str]:
    """Yield text fragments from a ``message.content`` value (string or
    multimodal list). Non-text parts (images, audio, …) are skipped."""
    if isinstance(content, str):
        if content:
            yield content
        return
    for part in _object_list(content) or ():
        if isinstance(part, str):
            # A bare string in a content/input list is itself a text
            # fragment (Responses-API mixed-list shape).
            if part:
                yield part
            continue
        fields = _text_object(part)
        if fields is None:
            continue
        if _is_json_part(fields):
            yield json.dumps(fields["data"], ensure_ascii=False)
            continue
        text = _part_text(fields)
        if text is not None:
            yield text


def _coerce_input_to_messages(input_value: object) -> list[dict[str, object]]:
    """Coerce a Responses-API ``data["input"]`` value into chat-style messages."""
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    if not isinstance(input_value, list):
        return []
    messages: Final[list[dict[str, object]]] = []
    for item in input_value:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
        elif isinstance(item, dict):
            if _part_text(item) is not None:
                messages.append({"role": item.get("role") or "user", "content": [item]})
            elif item.get("type") == "reasoning":
                if "content" in item:
                    messages.append(
                        {
                            "role": item.get("role") or "assistant",
                            "content": item["content"],
                        }
                    )
                if isinstance(item.get("summary"), list):
                    messages.append(
                        {
                            "role": item.get("role") or "assistant",
                            "content": item["summary"],
                        }
                    )
            elif "content" in item:
                messages.append({"role": item.get("role") or "user", "content": item["content"]})
            elif item.get("type") in _OUTPUT_ITEM_TYPES and "output" in item:
                messages.append({"role": item.get("role") or "tool", "content": item["output"]})
    return messages


def _iter_inspection_messages(data: Mapping[str, object]) -> Iterator[object]:
    """Yield every message-like dict, walking ``messages`` AND ``input``."""
    messages: Final = data.get("messages")
    if isinstance(messages, list):
        yield from messages
    yield from _coerce_input_to_messages(data.get("input"))


def iter_message_text(data: Mapping[str, object]) -> Iterator[str]:
    """Yield every text fragment from ``messages`` AND ``input``.

    Walks every role (user, assistant, system, …) — guardrails inspect
    the entire conversation, not just user turns.
    """
    for message in _iter_inspection_messages(data):
        if not isinstance(message, dict):
            continue
        yield from _iter_text_parts_in_content(message.get("content"))


def walk_user_text(
    data: dict[str, object],
    visit: Callable[[str], str],
    visit_json: Callable[[object], object] | None = None,
) -> int:
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
        parts: Final = _object_list(content)
        if parts is not None:
            new_parts: Final[list[object]] = []
            for part in parts:
                fields = _text_object(part)
                if isinstance(part, str) and part:
                    visited += 1
                    new_parts.append(visit(part))
                elif fields is not None and (text := _part_text(fields)) is not None:
                    visited += 1
                    new_parts.append({**fields, "text": visit(text)})
                elif (
                    fields is not None
                    and _is_json_part(fields)
                    and _json_has_text(fields["data"], with_keys=visit_json is not None)
                ):
                    visited += 1
                    payload = fields["data"]
                    new_parts.append(
                        {
                            **fields,
                            "data": visit_json(payload) if visit_json is not None else map_json_text(payload, visit),
                        }
                    )
                else:
                    new_parts.append(part)
            return new_parts
        return content

    messages: Final = data.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                message["content"] = _rewrite_content(message["content"])

    input_value: Final = data.get("input")
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
                if _part_text(item) is not None:
                    visited += 1
                    input_value[idx] = {**item, "text": visit(item["text"])}
                elif item.get("type") == "reasoning":
                    if "content" in item:
                        item["content"] = _rewrite_content(item["content"])
                    if isinstance(item.get("summary"), list):
                        item["summary"] = _rewrite_content(item["summary"])
                elif "content" in item:
                    item["content"] = _rewrite_content(item["content"])
                elif item.get("type") in _OUTPUT_ITEM_TYPES and "output" in item:
                    item["output"] = _rewrite_content(item["output"])
        return visited

    return visited


def is_string_batch_input(data: Mapping[str, object]) -> bool:
    """Return True when the only inspected content is an ``input`` list of plain
    strings, the /embeddings batch shape, which :func:`apply_redacted_messages_back`
    rewrites element-wise."""
    if "messages" in data:
        return False
    input_value: Final = data.get("input")
    return isinstance(input_value, list) and bool(input_value) and all(isinstance(item, str) for item in input_value)


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

    Returns False, leaving ``data`` untouched, when a batch response does not
    carry exactly one message per inspected element: a partial rewrite would
    forward the remaining originals unredacted. Callers must block on False.
    """
    if is_string_batch_input(data):
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
    if "messages" in data:
        data["messages"] = redacted_messages
    input_value: Final = data.get("input")
    if isinstance(input_value, str):
        text_parts: Final[list[str]] = []
        for msg in redacted_messages:
            if not isinstance(msg, dict):
                continue
            text_parts.extend(_iter_text_parts_in_content(msg.get("content")))
        data["input"] = "\n".join(text_parts)
    return True


def has_non_string_content(data: Mapping[str, object]) -> bool:
    """Return True if any inspected content is not a plain string.

    Used by hooks whose mask/redact path operates on string offsets and
    therefore cannot preserve multimodal non-text parts. Such hooks should
    degrade to block-on-detect when this returns True so image/audio parts
    are not silently stripped during in-place masking.
    """
    messages: Final = data.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and not isinstance(message.get("content"), str):
                if message.get("content") is not None:
                    return True
    input_value: Final = data.get("input")
    if input_value is not None and not isinstance(input_value, str):
        return True
    return False


def build_inspection_messages(data: dict[str, object]) -> list[dict[str, str]]:
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
        if not isinstance(message, dict):
            continue
        text = "\n".join(_iter_text_parts_in_content(message.get("content")))
        if not text:
            continue
        role = message.get("role", "user") or "user"
        flattened.append({"role": role, "content": text})
    return flattened
