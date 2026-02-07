"""
This plugin searches for Linear API Tokens and Linear Client Secrets.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class LinearDetector(RegexBasedDetector):
    """Scans for Linear secrets."""

    @property
    def secret_type(self) -> str:
        return "Linear Secret"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Linear API Token
            re.compile(r"""(?i)lin_api_[a-z0-9]{40}"""),
            # Linear Client Secret
            re.compile(
                r"""(?i)(?:linear)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-f0-9]{32})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
