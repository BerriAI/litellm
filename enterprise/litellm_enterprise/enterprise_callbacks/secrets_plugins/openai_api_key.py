"""
This plugin searches for OpenAI API Keys.
"""

import re
from collections.abc import Generator

from detect_secrets.plugins.base import RegexBasedDetector


class OpenAIApiKeyDetector(RegexBasedDetector):
    """Scans for OpenAI API Keys."""

    @property
    def secret_type(self) -> str:
        return "Strict OpenAI API Key"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"((?:(?<![a-zA-Z0-9])|(?<=%[0-9A-Fa-f]{2}))"
                r"sk[-_]"
                r"[a-zA-Z0-9_-]{5,}"
                r"(?![a-zA-Z0-9_-]))"
            )
        ]

    def analyze_string(self, string: str) -> Generator[str, None, None]:
        # the digit check lives outside the regex: a lookahead re-scans the token
        # from every `sk` inside it, which is quadratic on `-sk-sk-sk-...` input
        yield from (match for match in super().analyze_string(string) if re.search(r"[0-9]", match))
