"""
This plugin searches for Postman API Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class PostmanApiTokenDetector(RegexBasedDetector):
    """Scans for Postman API Tokens."""

    @property
    def secret_type(self) -> str:
        return "Postman API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"""(?i)\b(PMAK-[a-f0-9]{24}-[a-f0-9]{34})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            )
        ]
