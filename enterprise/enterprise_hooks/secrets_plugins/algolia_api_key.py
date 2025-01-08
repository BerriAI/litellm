"""
This plugin searches for Algolia API keys
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class AlgoliaApiKeyDetector(RegexBasedDetector):
    """Scans for Algolia API keys."""

    @property
    def secret_type(self) -> str:
        return "Algolia API Key"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(r"""(?i)\b((LTAI)[a-z0-9]{20})(?:['|\"|\n|\r|\s|\x60|;]|$)"""),
        ]
