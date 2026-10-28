"""
Replace messages with compact stubs and extract human-readable keys.
"""

import re
from typing import Final

from litellm.compression.content_detection import detect_content_type

# Patterns for extracting file paths from content
_FILE_PATH_PATTERNS: Final = [
    re.compile(r"^#\s*(\S+\.\w+)", re.MULTILINE),  # # filename.py
    re.compile(r"^//\s*(\S+\.\w+)", re.MULTILINE),  # // filename.js
    re.compile(r"^File:\s*(\S+)", re.MULTILINE),  # File: path/to/file
    re.compile(r"^---\s*(\S+\.\w+)", re.MULTILINE),  # --- filename.ext
    re.compile(r"`(\S+\.\w{1,5})`"),  # `filename.ext` in backticks
]


def extract_key(message: dict, fallback_index: int, used_keys: set[str]) -> str:
    """
    Extract a human-readable key for the message.

    Looks for file path patterns in the content. Falls back to message_{index}.
    Handles duplicates by appending _2, _3, etc.
    """
    content = message.get("content", "")
    if isinstance(content, list):
        content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)

    key = None
    for pattern in _FILE_PATH_PATTERNS:
        match = pattern.search(content[:2000])  # Only search the beginning
        if match:
            # Use just the filename, not full path
            path = match.group(1)
            key = path.split("/")[-1]
            break

    if key is None:
        key = f"message_{fallback_index}"

    # Handle duplicates
    base_key: Final = key
    counter = 2
    while key in used_keys:
        key = f"{base_key}_{counter}"
        counter += 1

    used_keys.add(key)
    return key


def stub_message(message: dict, key: str) -> dict:
    """
    Replace message content with a compact stub.

    Returns a new message dict with the same role but content replaced
    with a short description referencing the retrieval tool.
    """
    content = message.get("content", "")
    if isinstance(content, list):
        content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)

    line_count: Final = content.count("\n") + 1
    content_type: Final = detect_content_type(content)

    stub_content: Final = (
        f"[Compressed: {key} — {line_count} lines, {content_type}. "
        f"Use litellm_content_retrieve tool to get full content.]"
    )

    return {**message, "content": stub_content}


def truncate_message(message: dict, max_tokens: int) -> dict:
    """
    Truncate a message's content to approximately max_tokens by keeping
    the first 70% and last 30% of lines with a separator in between.

    Uses line-based splitting to preserve code structure (function
    boundaries, indentation) rather than word-based splitting which
    mangles code.

    Used when a message is too large to fit entirely in the budget but
    too relevant to fully stub out.
    """
    content = message.get("content", "")
    if isinstance(content, list):
        content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)

    # Rough conversion: 1 token ≈ 3 characters
    target_chars: Final = max(100, max_tokens * 3)

    if len(content) <= target_chars:
        return {**message, "content": content}

    lines: Final = content.split("\n")

    # Estimate target line count from character budget
    avg_line_len: Final = max(1, len(content) // max(1, len(lines)))
    target_lines: Final = max(2, target_chars // avg_line_len)

    if len(lines) <= target_lines:
        return {**message, "content": content}

    first_count: Final = (target_lines * 7) // 10
    last_count: Final = target_lines - first_count
    truncated: Final = (
        "\n".join(lines[:first_count]) + "\n...[truncated for context window]...\n" + "\n".join(lines[-last_count:])
    )
    return {**message, "content": truncated}
