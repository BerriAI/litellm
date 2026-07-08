"""Placeholder content for cleared ``tool_result`` blocks (string or block list)."""

from typing import Any, List, Union

from .constants import CLEARED_TOOL_RESULT_PLACEHOLDER


def build_cleared_tool_result_content(
    original_content: Any,
) -> Union[str, List[dict]]:
    """Return a string or single text block list, matching ``original_content`` shape."""
    if isinstance(original_content, list):
        return [{"type": "text", "text": CLEARED_TOOL_RESULT_PLACEHOLDER}]
    return CLEARED_TOOL_RESULT_PLACEHOLDER
