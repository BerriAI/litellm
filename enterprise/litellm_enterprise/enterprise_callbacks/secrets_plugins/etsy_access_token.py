"""
This plugin searches for Etsy Access Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class EtsyAccessTokenDetector(RegexBasedDetector):
    """Scans for Etsy Access Tokens."""

    @property
    def secret_type(self) -> str:
        return "Etsy Access Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Etsy Access Token
            re.compile(
                r"""(?i)(?:etsy)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{24})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
