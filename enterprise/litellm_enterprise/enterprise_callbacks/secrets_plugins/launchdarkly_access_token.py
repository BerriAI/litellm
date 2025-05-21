"""
This plugin searches for Launchdarkly Access Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class LaunchdarklyAccessTokenDetector(RegexBasedDetector):
    """Scans for Launchdarkly Access Tokens."""

    @property
    def secret_type(self) -> str:
        return "Launchdarkly Access Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"""(?i)(?:launchdarkly)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9=_\-]{40})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            )
        ]
