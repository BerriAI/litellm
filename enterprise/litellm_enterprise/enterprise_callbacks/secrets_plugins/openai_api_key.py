"""
This plugin searches for OpenAI API Keys.
"""

import re

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
                r"sk(?:-|_(?!live_))"
                r"(?=[a-zA-Z0-9_-]{5,}(?![a-zA-Z0-9_-]))"
                r"(?=[a-zA-Z0-9_-]*[0-9])"
                r"[a-zA-Z0-9_-]+(?![a-zA-Z0-9_-]))"
            )
        ]
