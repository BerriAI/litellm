"""
This plugin searches for GCP API keys.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class GCPApiKeyDetector(RegexBasedDetector):
    """Scans for GCP API keys."""

    @property
    def secret_type(self) -> str:
        return "GCP API Key"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # GCP API Key
            re.compile(
                r"""(?i)\b(AIza[0-9A-Za-z\\-_]{35})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
