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

from collections.abc import Mapping, Sequence
from typing import Final

from litellm.litellm_core_utils.prompt_templates.factory import get_attribute_or_key


def _as_dict(value: object) -> Mapping[object, object] | None:
    return value if isinstance(value, dict) else None


def _as_list(value: object) -> Sequence[object] | None:
    return value if isinstance(value, list) else None


def _field(source: object, name: str) -> object:
    return get_attribute_or_key(source, name, None)


def content_to_text(content: object) -> str:
    """Collapse a message ``content`` (str or list-of-parts) to plain text.

    For the multimodal list shape, joins ``{type: "text", text: ...}`` parts
    with blank-line separators; non-text parts are ignored.
    """
    if isinstance(content, str):
        return content
    items: Final = _as_list(content)
    if items is None:
        return ""
    parts: Final[list[str]] = []
    for item in items:
        part = _as_dict(item)
        if part is not None and part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n\n".join(parts)


def is_all_text_parts(content: object) -> bool:
    """True when ``content`` is a non-empty part list holding only text parts."""
    items: Final = _as_list(content)
    if not items:
        return False
    return all(part is not None and part.get("type") == "text" for part in map(_as_dict, items))


def merge_rewritten_text_parts(parts: Sequence[object], new_text: str) -> list[object]:
    """Collapse a rewritten all-text part list into one part carrying ``new_text``.

    Only all-text rows are ever flattened, so the merged part IS the whole row:
    it keeps the first part's fields and the LAST declared cache_control
    breakpoint. A breakpoint caches the prefix ending at its part, so after the
    merge the last one (and its TTL) is the one that still describes the row.
    """
    dict_parts: Final = tuple(part for part in map(_as_dict, parts) if part is not None)
    breakpoints: Final = tuple(part["cache_control"] for part in dict_parts if part.get("cache_control") is not None)
    base: Final[dict[object, object]] = (
        {**dict_parts[0], "text": new_text} if dict_parts else {"type": "text", "text": new_text}
    )
    return [{**base, "cache_control": breakpoints[-1]} if breakpoints else base]


def assistant_text_from_response(response: object) -> str | None:
    """The assistant's natural-language text from a model response, across chat,
    Anthropic, and Responses shapes. Preserved when the turn is rebuilt for the
    retrieval follow-up so the model's reasoning is not lost."""
    choices: Final = _as_list(_field(response, "choices"))
    if choices:
        message: Final = _field(choices[0], "message")
        if message is not None:
            text: Final = content_to_text(_field(message, "content"))
            if text:
                return text
    content: Final = _as_list(_field(response, "content"))
    if content is not None:
        parts: Final = [
            text
            for block in content
            if _field(block, "type") == "text"
            for text in (_field(block, "text"),)
            if isinstance(text, str) and text
        ]
        if parts:
            return "".join(parts)
    output: Final = _as_list(_field(response, "output"))
    if output is not None:
        output_parts: Final = [
            text
            for item in output
            if _field(item, "type") == "message"
            for chunk in (get_attribute_or_key(item, "content", None) or ())
            if _field(chunk, "type") == "output_text"
            for text in (_field(chunk, "text"),)
            if isinstance(text, str) and text
        ]
        if output_parts:
            return "".join(output_parts)
    return None
