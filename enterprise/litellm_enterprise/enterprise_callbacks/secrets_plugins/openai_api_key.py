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
        return [re.compile(r"""(sk-[a-zA-Z0-9]{5,})""")]
