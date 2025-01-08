"""
This plugin searches for RapidAPI Access Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class RapidApiAccessTokenDetector(RegexBasedDetector):
    """Scans for RapidAPI Access Tokens."""

    @property
    def secret_type(self) -> str:
        return "RapidAPI Access Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"""(?i)(?:rapidapi)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9_-]{50})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            )
        ]
