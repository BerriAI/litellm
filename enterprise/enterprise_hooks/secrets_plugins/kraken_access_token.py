"""
This plugin searches for Kraken Access Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class KrakenAccessTokenDetector(RegexBasedDetector):
    """Scans for Kraken Access Tokens."""

    @property
    def secret_type(self) -> str:
        return "Kraken Access Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Kraken Access Token
            re.compile(
                r"""(?i)(?:kraken)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9\/=_\+\-]{80,90})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
