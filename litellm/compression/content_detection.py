"""
Auto-detect content type per message: code, JSON, or text.
"""

import json
import re


_CODE_KEYWORDS = re.compile(
    r"\b(?:def |function |class |import |from |require\(|#include|fn |func |const |let |var |public |private |static )\b"
)


def detect_content_type(content: str) -> str:
    """
    Detect whether content is code, JSON, or plain text.

    Returns one of: "code", "json", "text"
    """
    stripped = content.strip()
    if not stripped:
        return "text"

    # Check JSON
    if stripped[0] in ("{", "["):
        try:
            json.loads(stripped)
            return "json"
        except (json.JSONDecodeError, ValueError):
            pass

    # Check code indicators
    # Sample first 5000 chars for performance
    sample = stripped[:5000]
    keyword_matches = len(_CODE_KEYWORDS.findall(sample))
    lines = sample.split("\n")
    indented_lines = sum(
        1 for line in lines if line.startswith(("    ", "\t")) and line.strip()
    )

    # If we see multiple code keywords or significant indentation, it's likely code
    if keyword_matches >= 3 or (indented_lines > len(lines) * 0.3 and len(lines) > 5):
        return "code"

    return "text"
