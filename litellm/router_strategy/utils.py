"""Shared utilities for router strategies."""
from typing import List, Optional, Union


def extract_text_from_input(input: Union[str, List]) -> Optional[str]:
    """
    Extract plain text from a Responses API ``input`` field.

    The Responses API accepts either a bare string or a list of input
    items (``ResponseInputParam``).  Each item may be:

    * A plain ``str``.
    * A dict with ``type="text"`` and a ``text`` key.
    * A dict with ``type="message"`` whose ``content`` is itself a string
      or a list of content parts (same ``{type, text}`` shape).

    Returns the concatenated text, or ``None`` when nothing extractable
    is found.
    """
    if isinstance(input, str):
        return input.strip() or None

    if not isinstance(input, list):
        return None

    parts: List[str] = []
    for item in input:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            item_type = item.get("type", "")
            if item_type == "text":
                text = item.get("text") or ""
                if text:
                    parts.append(text)
            elif item_type == "message":
                content = item.get("content") or ""
                if isinstance(content, str):
                    if content:
                        parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            t = part.get("text") or ""
                            if t:
                                parts.append(t)
    return " ".join(parts).strip() or None
