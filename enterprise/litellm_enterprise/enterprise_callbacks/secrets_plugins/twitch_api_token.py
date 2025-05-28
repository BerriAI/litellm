"""
This plugin searches for Twitch API Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class TwitchApiTokenDetector(RegexBasedDetector):
    """Scans for Twitch API Tokens."""

    @property
    def secret_type(self) -> str:
        return "Twitch API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"""(?i)(?:twitch)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{30})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            )
        ]
