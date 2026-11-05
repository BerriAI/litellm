import json
from typing import Any, Final  # noqa: TID251  # Existing annotations retained unchanged in this pure move

from pydantic import TypeAdapter

import litellm
from litellm import verbose_logger
from litellm.constants import REDACTED_BY_LITELLM
from litellm.litellm_core_utils.core_helpers import as_str_mapping

_DECODED_JSON: Final = TypeAdapter(object)
_ARGUMENT_PREVIEW_CHARS: Final = 200


def _logged_arguments(arguments: str) -> str:
    if litellm.turn_off_message_logging:
        return REDACTED_BY_LITELLM
    if len(arguments) <= _ARGUMENT_PREVIEW_CHARS:
        return arguments
    return f"{arguments[:_ARGUMENT_PREVIEW_CHARS]}..."


def _attempt_json_repair(s: str) -> object | None:
    """
    Attempt to repair truncated JSON produced by LLM tool calls.

    Handles the most common truncation patterns where the model generates
    valid JSON that is cut short (missing closing brackets/braces).

    Returns the parsed value on success, or None if repair fails.
    """
    import json

    stripped: Final = s.rstrip()
    if not stripped:
        return None

    # Track the stack of unmatched openers to respect nesting order
    opener_stack: Final[list] = []
    in_string = False
    escape_next = False

    for ch in stripped:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            opener_stack.append("}")
        elif ch == "[":
            opener_stack.append("]")
        elif ch in ("}", "]"):
            if opener_stack and opener_stack[-1] == ch:
                opener_stack.pop()

    if not opener_stack:
        return None

    # Remove trailing comma before we close brackets
    candidate = stripped.rstrip(",")

    # Close in reverse order of opening (respects nesting)
    candidate += "".join(reversed(opener_stack))

    try:
        return _DECODED_JSON.validate_python(json.loads(candidate))
    except json.JSONDecodeError:
        pass

    return None


def parse_tool_call_arguments(
    arguments: str | None,
    tool_name: str | None = None,
    context: str | None = None,
) -> Any:
    """
    Parse tool call arguments from a JSON string.

    When the JSON is malformed (e.g. truncated by the model), this function
    attempts a lightweight repair (closing unmatched brackets/braces) before
    raising an error.  A warning is logged whenever repair succeeds so that
    callers are aware the arguments were not perfectly formed.

    Args:
        arguments: The JSON string containing tool arguments, or None.
        tool_name: Optional name of the tool (for error messages).
        context: Optional context string (e.g., "Anthropic Messages API").

    Returns:
        Parsed arguments (usually a dict, but may be any JSON-deserializable
        type such as list, str, int, float, or None).  Returns empty dict if
        arguments is None or empty.

    Raises:
        ValueError: If the arguments string is not valid JSON and cannot be repaired.
    """
    import json

    if not arguments or not arguments.strip():
        return {}

    try:
        return _DECODED_JSON.validate_python(json.loads(arguments))
    except json.JSONDecodeError as original_error:
        repaired: Final = _attempt_json_repair(arguments)
        if repaired is not None:
            verbose_logger.warning(
                "Repaired truncated tool call arguments for tool '%s' (%s). Original (%d chars): %s",
                tool_name or "<unknown>",
                context or "unknown context",
                len(arguments),
                _logged_arguments(arguments),
            )
            return repaired

        error_parts: Final = ["Failed to parse tool call arguments"]

        if tool_name:
            error_parts.append(f"for tool '{tool_name}'")
        if context:
            error_parts.append(f"({context})")

        error_message: Final = " ".join(error_parts) + f". Error: {original_error}. Arguments: {arguments}"

        raise ValueError(error_message) from original_error


def split_concatenated_json_objects(raw: str) -> list[dict[str, object]]:
    """
    Split a string that contains one or more concatenated JSON objects into
    a list of parsed dicts.

    LLM providers (notably Bedrock Claude Sonnet 4.5) sometimes return
    multiple tool-call argument objects concatenated in a single
    ``arguments`` string, e.g.::

        '{"command":["curl",...]}{"command":["curl",...]}{"command":["curl",...]}'

    ``json.loads()`` fails on this with ``JSONDecodeError: Extra data``.
    This helper uses ``json.JSONDecoder.raw_decode()`` to walk the string
    and extract each JSON object individually.

    The walk degrades gracefully: if the string is malformed or truncated
    (e.g. a stream that ended mid-tool-call), whatever complete objects were
    parsed before the bad tail are returned and the remainder is discarded
    with a warning, rather than raising.  The sole caller
    (``_convert_to_bedrock_tool_call_invoke``) treats an empty result as
    ``input={}`` so the conversation can continue instead of hard-failing.

    Returns
    -------
    list[dict]
        A list of parsed dicts – one per JSON object found.  If *raw* is
        empty, whitespace-only, or wholly unparseable, an empty list is
        returned.
    """
    import json

    raw = raw.strip()
    if not raw:
        return []

    decoder: Final = json.JSONDecoder()
    results: Final[list[dict[str, object]]] = []
    idx = 0
    length: Final = len(raw)

    while idx < length:
        # Skip whitespace between objects
        while idx < length and raw[idx] in " \t\n\r":
            idx += 1
        if idx >= length:
            break

        try:
            decoded = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError as e:
            verbose_logger.warning(
                "split_concatenated_json_objects: discarding unparseable tool-call "
                "arguments tail after %d complete object(s); decode_start=%d error=%s",
                len(results),
                idx,
                e,
            )
            break
        obj = as_str_mapping(_DECODED_JSON.validate_python(decoded[0]))
        if obj is not None:
            results.append(dict(obj))
        else:
            # Non-dict JSON value – wrap in empty dict (Bedrock requires
            # toolUse.input to be an object).
            results.append({})
        idx = decoded[1]

    return results


MAX_SALVAGED_TOOL_ARGUMENT_OBJECTS: Final = 8


def salvage_concatenated_tool_arguments(raw: str) -> tuple[dict[str, object], ...]:
    """Return complete concatenated JSON objects that are safe to expand.

    Identical objects collapse to the first one and are not capped. More than
    ``MAX_SALVAGED_TOOL_ARGUMENT_OBJECTS`` objects that are not all identical
    returns an empty tuple. Anything that is not a full concatenation of JSON
    objects returns an empty tuple. Repeated copies of the first object are not
    retained, and once the cap is passed the rest of the string is only checked.
    """
    stripped: Final = raw.strip()
    if not stripped:
        return ()
    decoder: Final = json.JSONDecoder()
    length: Final = len(stripped)
    idx = 0  # rebind-ok: cursor walks the concatenated JSON string
    count = 0  # rebind-ok: counts complete objects without retaining duplicates
    kept = ()  # rebind-ok: holds at most one object past the salvage cap
    exceeded = False  # rebind-ok: cap already passed, the tail is only validated
    while idx < length:
        while idx < length and stripped[idx] in " \t\n\r":
            idx += 1
        if idx >= length:
            break
        try:
            decoded = decoder.raw_decode(stripped, idx)
        except json.JSONDecodeError:
            return ()
        decoded_object = as_str_mapping(_DECODED_JSON.validate_python(decoded[0]))
        if decoded_object is None:
            return ()
        idx = decoded[1]
        if exceeded:
            continue
        obj = dict(decoded_object)
        count += 1
        if not kept:
            kept = (obj,)
            continue
        if obj == kept[0] and len(kept) == 1:
            continue
        if len(kept) == 1 and count > 2 and count - 1 > MAX_SALVAGED_TOOL_ARGUMENT_OBJECTS:
            exceeded = True
            continue
        if len(kept) == 1 and count > 2:
            kept = (kept[0],) * (count - 1)
        if len(kept) >= MAX_SALVAGED_TOOL_ARGUMENT_OBJECTS:
            exceeded = True
            continue
        kept = (*kept, obj)
    if exceeded:
        return ()
    return kept


def tool_call_argument_count(arguments: str) -> int:
    """How many tool calls one raw argument string stands for, decided the way
    ``get_tool_calls_from_response`` decides it: valid or repairable JSON is one
    call, a concatenation of JSON objects is one call per salvaged object, and
    anything else is one call with empty arguments. Unlike that path it never
    logs any part of the argument text, so usage counters can call it on every
    response."""
    if not arguments.strip():
        return 1
    try:
        json.loads(arguments)
    except json.JSONDecodeError:
        if _attempt_json_repair(arguments) is not None:
            return 1
        return len(salvage_concatenated_tool_arguments(arguments)) or 1
    return 1
