"""
This plugin searches for Intercom API Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class IntercomApiTokenDetector(RegexBasedDetector):
    """Scans for Intercom API Tokens."""

    @property
    def secret_type(self) -> str:
        return "Intercom API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"""(?i)(?:intercom)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9=_\-]{60})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
