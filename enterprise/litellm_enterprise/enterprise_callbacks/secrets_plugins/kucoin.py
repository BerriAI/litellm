"""
This plugin searches for Kucoin Access Tokens and Secret Keys.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class KucoinDetector(RegexBasedDetector):
    """Scans for Kucoin Access Tokens and Secret Keys."""

    @property
    def secret_type(self) -> str:
        return "Kucoin Secret"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Kucoin Access Token
            re.compile(
                r"""(?i)(?:kucoin)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-f0-9]{24})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Kucoin Secret Key
            re.compile(
                r"""(?i)(?:kucoin)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
