"""
This plugin searches for Droneci Access Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class DroneciAccessTokenDetector(RegexBasedDetector):
    """Scans for Droneci Access Tokens."""

    @property
    def secret_type(self) -> str:
        return "Droneci Access Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Droneci Access Token
            re.compile(
                r"""(?i)(?:droneci)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{32})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
