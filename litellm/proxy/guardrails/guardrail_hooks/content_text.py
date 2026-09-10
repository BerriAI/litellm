"""Shared content-part helpers for compression guardrails (headroom, compresr).

Compression services only transform plain-string message content: every
transform in the service pipeline gates on ``isinstance(content, str)`` and
silently skips the OpenAI list-of-parts shape. Guardrails that send messages
to such a service collapse text-bearing part lists to strings here, and write
the rewritten text back through ``merge_rewritten_text_parts``.

Anthropic ``cache_control`` breakpoints are positional: each one caches the
prefix ending at the part that carries it. A single compressed string can
therefore only be written back over a run of text parts, never across a
non-text part, which is what ``is_all_text_parts`` gates.
"""

from collections.abc import Sequence
from typing import Final

from litellm.litellm_core_utils.prompt_templates.factory import get_attribute_or_key


def content_to_text(content: object) -> str:
    """Collapse a message ``content`` (str or list-of-parts) to plain text.

    For the multimodal list shape, joins ``{type: "text", text: ...}`` parts
    with blank-line separators; non-text parts are ignored.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: Final[list[str]] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n\n".join(parts)
    return ""


def is_all_text_parts(content: object) -> bool:
    """True when ``content`` is a non-empty part list holding only text parts."""
    if not isinstance(content, list) or not content:
        return False
    return all(isinstance(part, dict) and part.get("type") == "text" for part in content)


def merge_rewritten_text_parts(parts: Sequence[object], new_text: str) -> list[object]:
    """Collapse a rewritten all-text part list into one part carrying ``new_text``.

    Only all-text rows are ever flattened, so the merged part IS the whole row:
    it keeps the first part's fields and the LAST declared cache_control
    breakpoint. A breakpoint caches the prefix ending at its part, so after the
    merge the last one (and its TTL) is the one that still describes the row.
    """
    dict_parts: Final = tuple(part for part in parts if isinstance(part, dict))
    breakpoints: Final = tuple(part["cache_control"] for part in dict_parts if part.get("cache_control") is not None)
    base: Final = {**dict_parts[0], "text": new_text} if dict_parts else {"type": "text", "text": new_text}
    return [{**base, "cache_control": breakpoints[-1]} if breakpoints else base]


def assistant_text_from_response(response: object) -> str | None:
    """The assistant's natural-language text from a model response, across chat,
    Anthropic, and Responses shapes. Preserved when the turn is rebuilt for the
    retrieval follow-up so the model's reasoning is not lost."""
    choices: Final = get_attribute_or_key(response, "choices", None)
    if isinstance(choices, list) and choices:
        message: Final = get_attribute_or_key(choices[0], "message", None)
        if message is not None:
            text: Final = content_to_text(get_attribute_or_key(message, "content", None))
            if text:
                return text
    content: Final = get_attribute_or_key(response, "content", None)
    if isinstance(content, list):
        parts: Final = [
            text
            for block in content
            if get_attribute_or_key(block, "type", None) == "text"
            for text in (get_attribute_or_key(block, "text", None),)
            if isinstance(text, str) and text
        ]
        if parts:
            return "".join(parts)
    output: Final = get_attribute_or_key(response, "output", None)
    if isinstance(output, list):
        output_parts: Final = [
            text
            for item in output
            if get_attribute_or_key(item, "type", None) == "message"
            for chunk in (get_attribute_or_key(item, "content", None) or ())
            if get_attribute_or_key(chunk, "type", None) == "output_text"
            for text in (get_attribute_or_key(chunk, "text", None),)
            if isinstance(text, str) and text
        ]
        if output_parts:
            return "".join(output_parts)
    return None
