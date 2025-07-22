"""
This plugin searches for Readme API Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class ReadmeApiTokenDetector(RegexBasedDetector):
    """Scans for Readme API Tokens."""

    @property
    def secret_type(self) -> str:
        return "Readme API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(r"""(?i)\b(rdme_[a-z0-9]{70})(?:['|\"|\n|\r|\s|\x60|;]|$)""")
        ]
