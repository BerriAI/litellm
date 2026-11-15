"""
This plugin searches for Sentry Access Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class SentryAccessTokenDetector(RegexBasedDetector):
    """Scans for Sentry Access Tokens."""

    @property
    def secret_type(self) -> str:
        return "Sentry Access Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"""(?i)(?:sentry)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-f0-9]{64})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            )
        ]
