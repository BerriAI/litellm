"""
Replace messages with compact stubs and extract human-readable keys.
"""

import re
from typing import Set

from litellm.compression.content_detection import detect_content_type

# Patterns for extracting file paths from content
_FILE_PATH_PATTERNS = [
    re.compile(r"^#\s*(\S+\.\w+)", re.MULTILINE),  # # filename.py
    re.compile(r"^//\s*(\S+\.\w+)", re.MULTILINE),  # // filename.js
    re.compile(r"^File:\s*(\S+)", re.MULTILINE),  # File: path/to/file
    re.compile(r"^---\s*(\S+\.\w+)", re.MULTILINE),  # --- filename.ext
    re.compile(r"`(\S+\.\w{1,5})`"),  # `filename.ext` in backticks
]


def extract_key(message: dict, fallback_index: int, used_keys: Set[str]) -> str:
    """
    Extract a human-readable key for the message.

    Looks for file path patterns in the content. Falls back to message_{index}.
    Handles duplicates by appending _2, _3, etc.
    """
    content = message.get("content", "")
    if isinstance(content, list):
        content = " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )

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
    base_key = key
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
        content = " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )

    line_count = content.count("\n") + 1
    content_type = detect_content_type(content)

    stub_content = (
        f"[Compressed: {key} — {line_count} lines, {content_type}. "
        f"Use litellm_content_retrieve tool to get full content.]"
    )

    return {**message, "content": stub_content}


def truncate_message(message: dict, max_tokens: int) -> dict:
    """
    Truncate a message's content to approximately max_tokens by keeping
    the first 70% and last 30% of words with a separator in between.

    Used when a message is too large to fit entirely in the budget but
    too relevant to fully stub out.
    """
    content = message.get("content", "")
    if isinstance(content, list):
        content = " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )

    # Rough conversion: 1 token ≈ 0.75 words
    target_words = max(1, int(max_tokens * 0.75))
    words = content.split()

    if len(words) <= target_words:
        return {**message, "content": content}

    first_count = (target_words * 2) // 3
    last_count = target_words - first_count
    truncated = (
        " ".join(words[:first_count])
        + "\n...[truncated for context window]...\n"
        + " ".join(words[-last_count:])
    )
    return {**message, "content": truncated}
