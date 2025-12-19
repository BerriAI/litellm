"""
Generic tool call token cleaning for Bedrock models.

Some Bedrock models return tool calls as proprietary tokens embedded in text
rather than structured toolUse blocks. This module detects and cleans those tokens.

Design: Auto-detect pattern from response content, fully generic.
"""

import re
from typing import Dict, Optional


# =============================================================================
# Pattern Definitions - Add new patterns here as models are discovered
# =============================================================================

TOOL_CALL_PATTERNS: Dict[str, Dict[str, str]] = {
    # Pipe-style markers: <|marker|> format (used by Kimi-K2-Thinking)
    "pipe_markers": {
        "begin": "<|tool_call_begin|>",
        "arg_begin": "<|tool_call_argument_begin|>",
        "end": "<|tool_call_end|>",
        "section_begin": "<|tool_calls_section_begin|>",
        "section_end": "<|tool_calls_section_end|>",
    },
    # Add more patterns here as discovered from other models
}


# =============================================================================
# Auto-Detection
# =============================================================================


def detect_pattern_in_text(text: str) -> Optional[str]:
    """
    Auto-detect which tool call pattern is present in text.

    Args:
        text: Text content to check

    Returns:
        Pattern name if detected, None otherwise
    """
    for pattern_name, markers in TOOL_CALL_PATTERNS.items():
        if markers["begin"] in text:
            return pattern_name
        if "section_begin" in markers and markers["section_begin"] in text:
            return pattern_name
    return None


# =============================================================================
# Core Cleaning Function
# =============================================================================


def clean_tool_tokens_from_text(text: str, model: str = "") -> str:
    """
    Remove proprietary tool call tokens from text content.

    Auto-detects the pattern from the text itself.

    Args:
        text: The raw text content from model response
        model: Optional (unused, kept for API compatibility)

    Returns:
        Cleaned text with proprietary tokens removed
    """
    pattern_name = detect_pattern_in_text(text)
    if pattern_name is None:
        return text

    markers = TOOL_CALL_PATTERNS[pattern_name]
    cleaned = text

    # Remove section wrappers if present
    if "section_begin" in markers and markers["section_begin"] in cleaned:
        section_end = markers.get("section_end", "")
        if section_end:
            section_pattern = (
                rf"{re.escape(markers['section_begin'])}"
                rf".*?"
                rf"{re.escape(section_end)}"
            )
            cleaned = re.sub(section_pattern, "", cleaned, flags=re.DOTALL)

    # Remove individual tool call blocks
    if "arg_begin" in markers:
        # Pattern with argument marker: <begin>...<arg_begin>...<end>
        tool_pattern = (
            rf"{re.escape(markers['begin'])}"
            rf"[^<]*"
            rf"{re.escape(markers['arg_begin'])}"
            rf"[^<]*"
            rf"{re.escape(markers['end'])}"
        )
    else:
        # Simple begin/end pattern: <begin>...<end>
        tool_pattern = (
            rf"{re.escape(markers['begin'])}" rf".*?" rf"{re.escape(markers['end'])}"
        )

    cleaned = re.sub(tool_pattern, "", cleaned, flags=re.DOTALL)

    # Clean up whitespace
    cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)
    return cleaned.strip()
