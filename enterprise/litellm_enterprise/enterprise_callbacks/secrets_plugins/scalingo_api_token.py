"""
This plugin searches for Scalingo API Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class ScalingoApiTokenDetector(RegexBasedDetector):
    """Scans for Scalingo API Tokens."""

    @property
    def secret_type(self) -> str:
        return "Scalingo API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [re.compile(r"""\btk-us-[a-zA-Z0-9-_]{48}\b""")]
