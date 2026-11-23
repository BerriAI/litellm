"""Text a coding agent adds to a conversation that its user never typed.

Claude Code wraps project instructions, memory and status notes in ``<system-reminder>`` blocks inside the user's
own message. Codex sends its AGENTS.md and environment as separate user messages in instruction envelopes.
"""

import string
from collections.abc import Iterator, Mapping, Sequence
from itertools import accumulate
from typing import Final, TypeAlias

from pydantic import TypeAdapter, ValidationError

MarkerPairs: TypeAlias = tuple[tuple[str, str], ...]

SYSTEM_REMINDER_MARKERS: Final[MarkerPairs] = (("<system-reminder>", "</system-reminder>"),)
CODEX_MARKERS: Final[MarkerPairs] = SYSTEM_REMINDER_MARKERS + (
    ("<environment_context>", "</environment_context>"),
    ("<recommended_plugins>", "</recommended_plugins>"),
    ("<user_instructions>", "</user_instructions>"),
    ("<environments_instructions>", "</environments_instructions>"),
    ("# agents.md instructions for ", "</instructions>"),
)
OPENCODE_MARKERS: Final[MarkerPairs] = SYSTEM_REMINDER_MARKERS + (
    ("<task_result>", "</task_result>"),
    ("<subagent ", "</subagent>"),
)
_ASCII_LOWER: Final = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)
_TEXT_PART_TYPES: Final = frozenset({"text", "input_text"})
_INSTRUCTION_ROLES: Final = frozenset({"user", "developer", "system"})
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_JSON_ARRAY: Final = TypeAdapter(list[object])


def _block_spans(lowered: str, open_marker: str, close_marker: str) -> Iterator[tuple[int, int]]:
    """Span of each complete block for one marker pair, left to right.

    Literal `str.find`, not a regex: the delimiters are fixed strings, and `<system-reminder>.*?`
    retried its lazy quantifier from every opening tag, so repeated unclosed tags were quadratic
    (272KB took 7.6s) on input any keyholder controls. The cursor only moves forward and an unclosed
    tag ends the scan, so this is linear without bounding the input.
    """
    cursor = 0
    while (start := lowered.find(open_marker, cursor)) != -1:
        end = lowered.find(close_marker, start + len(open_marker))
        if end == -1:
            return
        cursor = end + len(close_marker)
        yield start, cursor


def _kept_segments(text: str, marker_pairs: MarkerPairs) -> tuple[str, ...] | None:
    """The stripped text around every complete marked block, or None when text has no complete block.

    Blocks from different pairs can nest or overlap, which the gap construction below would
    otherwise mishandle: an inner block's end would resume the kept text partway through the outer
    block, leaking the rest of that block. Running the block ends through a maximum resumes each gap
    past the furthest block seen so far, which collapses nested and overlapping spans without a
    separate merge pass. A single pair's ends already increase, so the maximum is the identity there.

    Deliberately linear in both the text and the block count, since both a regex scan and a fold
    that rebuilds a growing tuple of merged spans go quadratic on inputs that are cheap to send.
    Only ASCII is lowered for matching, because `str.lower` lengthens some characters (``İ``) and
    would shift every span after them.
    """
    lowered: Final = text.translate(_ASCII_LOWER)
    spans: Final = tuple(
        sorted(
            span
            for open_marker, close_marker in marker_pairs
            for span in _block_spans(lowered, open_marker, close_marker)
        )
    )
    if not spans:
        return None
    keep_from: Final = (0, *accumulate((end for _, end in spans), max))
    keep_to: Final = (*(start for start, _ in spans), len(text))
    return tuple(kept for a, b in zip(keep_from, keep_to) if (kept := text[a:b].strip()))


def strip_marked_blocks(text: str, marker_pairs: MarkerPairs = SYSTEM_REMINDER_MARKERS) -> str:
    """Text without its complete marked blocks, as one stripped line of everything written around them."""
    kept: Final = _kept_segments(text, marker_pairs)
    return text.strip() if kept is None else " ".join(kept)


def without_marked_blocks(text: str, marker_pairs: MarkerPairs) -> str:
    """Text without its complete marked blocks, left byte for byte as it was when it has none."""
    kept: Final = _kept_segments(text, marker_pairs)
    return text if kept is None else "\n\n".join(kept)


def _json_object(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, dict):
        return None
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


def _json_array(value: object) -> Sequence[object] | None:
    return _JSON_ARRAY.validate_python(value) if isinstance(value, list) else None


def _without_blocks_in_content(content: object, marker_pairs: MarkerPairs) -> object:
    if isinstance(content, str):
        return without_marked_blocks(content, marker_pairs)
    items: Final = _json_array(content)
    if items is None:
        return content
    parts: Final = (_without_blocks_in_part(part, marker_pairs) for part in items)
    return [part for part in parts if part is not None]


def _without_blocks_in_part(part: object, marker_pairs: MarkerPairs) -> object | None:
    """The content part without marked blocks, or None when it held nothing else."""
    fields: Final = _json_object(part)
    if fields is None:
        return part
    text: Final = fields.get("text")
    if fields.get("type") in _TEXT_PART_TYPES and isinstance(text, str):
        kept: Final = without_marked_blocks(text, marker_pairs)
        return {**fields, "text": kept} if kept or not text else None
    if fields.get("type") == "tool_result" and "content" in fields:
        return {**fields, "content": _without_blocks_in_content(fields["content"], marker_pairs)}
    return part


def _without_blocks_in_message(message: object, marker_pairs: MarkerPairs) -> object | None:
    """The message without marked blocks, or None when a user turn held nothing else."""
    fields: Final = _json_object(message)
    if fields is None:
        return message
    output: Final = fields.get("output")
    if fields.get("type") == "function_call_output" and isinstance(output, str):
        return {**fields, "output": without_marked_blocks(output, marker_pairs)}
    if "content" not in fields:
        return message
    content: Final = fields["content"]
    kept: Final = _without_blocks_in_content(content, marker_pairs)
    emptied: Final = bool(content) and not kept
    return None if emptied and fields.get("role") in _INSTRUCTION_ROLES else {**fields, "content": kept}


def without_harness_context(messages: object, marker_pairs: MarkerPairs) -> object:
    """A conversation as its user wrote it: every complete marked block removed from every message and tool
    result, and a user turn the agent sent that held only such blocks dropped. Anything else stays as it was."""
    if isinstance(messages, str):
        return without_marked_blocks(messages, marker_pairs)
    items: Final = _json_array(messages)
    if items is None:
        return messages
    kept: Final = (_without_blocks_in_message(message, marker_pairs) for message in items)
    return [message for message in kept if message is not None]
