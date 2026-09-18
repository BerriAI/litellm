"""
This plugin searches for Typeform API Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class TypeformApiTokenDetector(RegexBasedDetector):
    """Scans for Typeform API Tokens."""

    @property
    def secret_type(self) -> str:
        return "Typeform API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"""(?i)(?:typeform)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}(tfp_[a-z0-9\-_\.=]{59})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            )
        ]
