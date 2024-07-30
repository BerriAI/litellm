"""
This plugin searches for Facebook Access Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class FacebookAccessTokenDetector(RegexBasedDetector):
    """Scans for Facebook Access Tokens."""

    @property
    def secret_type(self) -> str:
        return "Facebook Access Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Facebook Access Token
            re.compile(
                r"""(?i)(?:facebook)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-f0-9]{32})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
