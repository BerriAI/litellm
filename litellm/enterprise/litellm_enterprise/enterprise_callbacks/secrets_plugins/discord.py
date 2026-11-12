"""
This plugin searches for Discord Client tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class DiscordDetector(RegexBasedDetector):
    """Scans for various Discord Client Tokens."""

    @property
    def secret_type(self) -> str:
        return "Discord Client Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Discord API key
            re.compile(
                r"""(?i)(?:discord)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-f0-9]{64})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Discord client ID
            re.compile(
                r"""(?i)(?:discord)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([0-9]{18})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Discord client secret
            re.compile(
                r"""(?i)(?:discord)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9=_\-]{32})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
