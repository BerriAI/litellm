"""
This plugin searches for LinkedIn Client IDs and LinkedIn Client secrets.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class LinkedInDetector(RegexBasedDetector):
    """Scans for LinkedIn secrets."""

    @property
    def secret_type(self) -> str:
        return "LinkedIn Secret"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # LinkedIn Client ID
            re.compile(
                r"""(?i)(?:linkedin|linked-in)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{14})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # LinkedIn Client secret
            re.compile(
                r"""(?i)(?:linkedin|linked-in)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{16})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
