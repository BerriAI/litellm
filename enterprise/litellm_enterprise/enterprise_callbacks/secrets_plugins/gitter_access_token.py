"""
This plugin searches for Gitter Access Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class GitterAccessTokenDetector(RegexBasedDetector):
    """Scans for Gitter Access Tokens."""

    @property
    def secret_type(self) -> str:
        return "Gitter Access Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Gitter Access Token
            re.compile(
                r"""(?i)(?:gitter)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9_-]{40})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
