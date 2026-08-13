"""
This plugin searches for Sendbird Access IDs and Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class SendbirdDetector(RegexBasedDetector):
    """Scans for Sendbird Access IDs and Tokens."""

    @property
    def secret_type(self) -> str:
        return "Sendbird Credential"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Sendbird Access ID
            re.compile(
                r"""(?i)(?:sendbird)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Sendbird Access Token
            re.compile(
                r"""(?i)(?:sendbird)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-f0-9]{40})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
