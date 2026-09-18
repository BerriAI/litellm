"""
This plugin searches for Snyk API Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class SnykApiTokenDetector(RegexBasedDetector):
    """Scans for Snyk API Tokens."""

    @property
    def secret_type(self) -> str:
        return "Snyk API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"""(?i)(?:snyk)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            )
        ]
