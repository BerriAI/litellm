"""
Shared helpers for guardrail hooks: extract text from a request body
regardless of whether it uses Chat Completions ``messages``, Responses-API
``input``, or multimodal list-format ``content`` parts.

Hooks that only check ``data["messages"]`` for string content silently
skip the other shapes — these helpers normalise that so every hook sees
every text fragment.
"""

import json
import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import JsonValue, TypeAdapter

from litellm.litellm_core_utils.core_helpers import iter_items
from litellm.responses.litellm_completion_transformation.compaction_items import (
    compaction_summary,
    input_items,
    with_visited_compaction_summary,
)

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


def _tool_json_object(pairs: Sequence[tuple[str, JsonValue]]) -> JsonValue:
    value: Final = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("Duplicate tool argument keys")
    return value


def _finite_number(literal: str) -> float:
    number: Final = float(literal)
    if not math.isfinite(number):
        raise ValueError("Non-finite tool argument number")
    return number


def _tool_arguments(arguments: str) -> JsonValue:
    parsed: Final = _tool_json(
        json.loads(
            arguments,
            object_pairs_hook=_tool_json_object,
            parse_float=_finite_number,
            parse_constant=_finite_number,
        )
    )
    json.dumps(parsed, ensure_ascii=False).encode("utf-8")  # raises for lone surrogates, which stay raw text
    return parsed


_JSON_STRING_TOKEN: Final = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"')

_KEY_SUFFIX: Final = re.compile(r"[ \t\n\r]*:")


def _value_string_spans(arguments: str) -> Iterator[tuple[int, int]]:
    """Yield the span of every non-empty string value in the valid JSON ``arguments``, in document order."""
    for token in _JSON_STRING_TOKEN.finditer(arguments):
        if token.end() - token.start() > 2 and _KEY_SUFFIX.match(arguments, token.end()) is None:
            yield token.span()


def _respliced(arguments: str, parsed: JsonValue, rewritten: JsonValue) -> str:
    """Serialize ``rewritten`` by replacing only its changed string values in ``arguments``, when nothing else changed."""
    serialized: Final = json.dumps(rewritten, ensure_ascii=False)
    strings: Final = iter(tuple(iter_tool_json_text(rewritten)))
    if json.dumps(map_tool_json_text(parsed, lambda text: next(strings, text)), ensure_ascii=False) != serialized:
        return serialized
    values: Final = zip(_value_string_spans(arguments), iter_tool_json_text(parsed), iter_tool_json_text(rewritten))
    changed: Final = tuple((span, new) for span, old, new in values if new != old)
    starts: Final = (0, *(end for (_, end), _ in changed))
    ends: Final = (*(start for (start, _), _ in changed), len(arguments))
    replacements: Final = (*(json.dumps(new, ensure_ascii=False) for _, new in changed), "")
    return "".join(arguments[start:end] + text for start, end, text in zip(starts, ends, replacements))


def iter_tool_json_fields(value: object) -> Iterator[tuple[str | None, str]]:
    """Yield ``(nearest key, text)`` for every non-empty string leaf; list items take their parent's key."""
    pending: Final[list[tuple[object, str | None]]] = [(value, None)]  # mutable-ok: owned DFS stack avoids recursion
    while pending:
        current, key = pending.pop()
        if isinstance(current, str):
            if current:
                yield key, current
            continue
        if current is None or isinstance(current, (bool, int, float)):
            continue
        fields = _text_object(current)
        if fields is not None:
            pending.extend((child, child_key) for child_key, child in reversed(tuple(fields.items())))
            continue
        items = _object_list(current)
        if items is None:
            raise ValueError("Invalid tool JSON payload")
        pending.extend((child, key) for child in reversed(items))


def iter_tool_json_text(value: object) -> Iterator[str]:
    for _, text in iter_tool_json_fields(value):
        yield text


@dataclass(frozen=True, slots=True)
class _ToolFrame:
    source: object
    parent: dict[str, JsonValue] | list[JsonValue]  # mutable-ok: frame writes only into the mapper's fresh output tree
    key: str | int

    def write(self, value: JsonValue) -> None:
        if isinstance(self.parent, dict) and isinstance(self.key, str):
            self.parent[self.key] = value
            return
        if isinstance(self.parent, list) and isinstance(self.key, int):
            self.parent[self.key] = value
            return
        raise ValueError("Invalid tool JSON payload")


def map_tool_json_text(value: object, visit: Callable[[str], str]) -> JsonValue:
    root: Final[list[JsonValue]] = [None]  # mutable-ok: one owned root slot receives the iterative result
    pending: Final[list[_ToolFrame]] = [_ToolFrame(value, root, 0)]  # mutable-ok: owned traversal stack
    while pending:
        frame = pending.pop()
        current = frame.source
        if isinstance(current, str):
            frame.write(visit(current) if current else current)
            continue
        if current is None or isinstance(current, (bool, int, float)):
            frame.write(current)
            continue
        fields = _text_object(current)
        if fields is not None:
            mapped: dict[str, JsonValue] = {key: None for key in fields}  # mutable-ok: owned output slots
            frame.write(mapped)
            pending.extend(_ToolFrame(child, mapped, key) for key, child in reversed(tuple(fields.items())))
            continue
        children = _object_list(current)
        if children is None:
            raise ValueError("Invalid tool JSON payload")
        items: list[JsonValue] = [None] * len(children)  # mutable-ok: fresh indexed output slots
        frame.write(items)
        pending.extend(_ToolFrame(child, items, index) for index, child in reversed(tuple(enumerate(children))))
    return root[0]


def _tool_json(value: object) -> JsonValue:
    return map_tool_json_text(value, lambda text: text)


# Responses-API item types whose ``output`` field carries user/tool text
# that guardrails should inspect.  ``function_call_output`` is the
# built-in shape; ``custom_tool_call_output`` is the custom-tool
# counterpart (see ``ChatCompletionCustomToolCallOutput``).
_OUTPUT_ITEM_TYPES: Final[frozenset[str]] = frozenset({"function_call_output", "custom_tool_call_output"})
# Responses input items carrying the model's tool call: ``function_call`` holds
# JSON ``arguments`` and ``custom_tool_call`` holds a freeform ``input`` string.
_CALL_ITEM_FIELDS: Final[Mapping[str, str]] = {"function_call": "arguments", "custom_tool_call": "input"}


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
            text = _part_text(part)
            if text is not None:
                yield text


def _coerce_input_to_messages(input_value: object) -> list[dict[str, object]]:
    """Coerce a Responses-API ``data["input"]`` value into chat-style messages."""
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    messages: Final[list[dict[str, object]]] = []
    for item in input_items(input_value):
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
        elif (summary := compaction_summary(item)) is not None:
            messages.append({"role": "user", "content": summary})
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


def _call_item(item: object) -> tuple[str, object] | None:
    """Return ``(field, value)`` for a Responses tool-call input item, else None."""
    fields: Final = _text_object(item)
    field: Final = _CALL_ITEM_FIELDS.get(str(fields.get("type"))) if fields is not None else None
    return (field, fields.get(field)) if fields is not None and field is not None and field in fields else None


def walk_user_text(
    data: dict[str, object],
    visit: Callable[[str], str],
    visit_tool: Callable[[JsonValue], JsonValue] | None = None,
) -> int:
    """Rewrite every text fragment in place via ``visit``.

    Mutates ``data["messages"]`` and ``data["input"]`` (including tool payloads), ``data["system"]`` and
    ``data["instructions"]``. Returns the number
    of fragments visited so callers can short-circuit when nothing was
    inspected.
    """
    visited = 0

    def _rewrite_tool(value: object) -> JsonValue:
        nonlocal visited
        payload: Final = _tool_json(value)
        if not tuple(iter_tool_json_text(payload)):
            return payload
        visited += 1  # rebind-ok: tally the structured payload inspected by this callback
        return visit_tool(payload) if visit_tool is not None else map_tool_json_text(payload, visit)

    def _rewrite_arguments(arguments: object) -> object:
        nonlocal visited
        if isinstance(arguments, dict):
            return _rewrite_tool(arguments)
        if not isinstance(arguments, str) or not arguments:
            return arguments
        try:
            parsed: Final = _tool_arguments(arguments)
        except (ValueError, RecursionError):
            visited += 1  # rebind-ok: malformed arguments still undergo the text callback
            return visit(arguments)
        if not isinstance(parsed, (dict, list)):
            visited += 1  # rebind-ok: scalar arguments still undergo the text callback
            return visit(arguments)
        rewritten: Final = _rewrite_tool(parsed)
        return arguments if rewritten == parsed else _respliced(arguments, parsed, rewritten)

    def _rewrite_part(part: Mapping[str, object], native: bool) -> Mapping[str, object]:
        nonlocal visited
        text: Final = _part_text(part)
        rewritten: Final = {**part, **({"text": visit(text)} if text is not None else {})}
        if text is not None:
            visited += 1  # rebind-ok: tally each existing text field alongside the native payload
        if native and part.get("type") == "tool_result" and "content" in part:
            return {**rewritten, "content": _rewrite_content(part["content"], native=False)}
        if native and part.get("type") == "tool_use" and "input" in part:
            return {**rewritten, "input": _rewrite_tool(part["input"])}
        return rewritten

    def _rewrite_content(content: object, native: bool = True) -> object:
        nonlocal visited
        if isinstance(content, str):
            if content:
                visited += 1
                return visit(content)
            return content
        if isinstance(content, list):
            new_parts: Final[list[object]] = []
            for part in content:
                if isinstance(part, str) and part:
                    visited += 1
                    new_parts.append(visit(part))
                elif isinstance(part, dict):
                    new_parts.append(_rewrite_part(part, native))
                else:
                    new_parts.append(part)
            return new_parts
        return content

    messages: Final = data.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if "content" in message:
                message["content"] = _rewrite_content(message["content"])
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    function = call.get("function") if isinstance(call, dict) else None
                    if isinstance(function, dict) and "arguments" in function:
                        function["arguments"] = _rewrite_arguments(function["arguments"])

    if "system" in data:
        data["system"] = _rewrite_content(data["system"])  # rebind-ok: preserve the native hook's in-place contract
    if "instructions" in data:
        data["instructions"] = _rewrite_content(data["instructions"])  # rebind-ok: Responses system prompt, in place

    input_value: Final = data.get("input")
    if isinstance(input_value, str):
        if input_value:
            visited += 1
            data["input"] = visit(input_value)
        return visited
    if isinstance(input_value, list):
        for idx, item in enumerate(input_items(input_value)):
            if isinstance(item, str):
                if item:
                    visited += 1
                    input_value[idx] = visit(item)
            elif (compaction := with_visited_compaction_summary(item, visit)) is not None:
                visited += 1
                input_value[idx] = compaction
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
                elif (call := _call_item(item)) is not None:
                    item[call[0]] = _rewrite_arguments(call[1])
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


def apply_redacted_messages_back(data: dict[str, object], redacted_messages: Sequence[object]) -> bool:
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
    input_value: Final = data.get("input")
    if is_string_batch_input(data):
        inspected_indices: Final = tuple(idx for idx, item in enumerate(iter_items(input_value)) if item)
        if len(redacted_messages) != len(inspected_indices):
            return False
        if any(not isinstance(message, Mapping) or message.get("content") is None for message in redacted_messages):
            return False
        redacted_texts: Final = tuple(
            "\n".join(_iter_text_parts_in_content(message["content"])) for message in redacted_messages
        )
        if isinstance(input_value, list):
            for idx, text in zip(inspected_indices, redacted_texts):
                input_value[idx] = text
        return True
    if "messages" in data:
        data["messages"] = redacted_messages
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
