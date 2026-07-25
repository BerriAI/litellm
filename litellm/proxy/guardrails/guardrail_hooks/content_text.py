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


def content_to_text(content: object) -> str:
    """Collapse a message ``content`` (str or list-of-parts) to plain text.

    For the multimodal list shape, joins ``{type: "text", text: ...}`` parts
    with blank-line separators; non-text parts are ignored.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
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
    dict_parts = tuple(part for part in parts if isinstance(part, dict))
    breakpoints = tuple(part["cache_control"] for part in dict_parts if part.get("cache_control") is not None)
    base = {**dict_parts[0], "text": new_text} if dict_parts else {"type": "text", "text": new_text}
    return [{**base, "cache_control": breakpoints[-1]} if breakpoints else base]
