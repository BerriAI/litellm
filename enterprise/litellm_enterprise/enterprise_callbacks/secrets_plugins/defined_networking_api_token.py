"""
This plugin searches for Defined Networking API Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class DefinedNetworkingApiTokenDetector(RegexBasedDetector):
    """Scans for Defined Networking API Tokens."""

    @property
    def secret_type(self) -> str:
        return "Defined Networking API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"""(?i)(?:dnkey)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}(dnkey-[a-z0-9=_\-]{26}-[a-z0-9=_\-]{52})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
