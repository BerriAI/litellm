"""
This plugin searches for Okta Access Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class OktaAccessTokenDetector(RegexBasedDetector):
    """Scans for Okta Access Tokens."""

    @property
    def secret_type(self) -> str:
        return "Okta Access Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"""(?i)(?:okta)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9=_\-]{42})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            )
        ]
