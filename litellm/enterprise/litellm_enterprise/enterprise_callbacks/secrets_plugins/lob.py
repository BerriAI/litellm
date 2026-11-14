"""
This plugin searches for Lob API secrets and Lob Publishable API keys.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class LobDetector(RegexBasedDetector):
    """Scans for Lob secrets."""

    @property
    def secret_type(self) -> str:
        return "Lob Secret"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Lob API Key
            re.compile(
                r"""(?i)(?:lob)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}((live|test)_[a-f0-9]{35})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Lob Publishable API Key
            re.compile(
                r"""(?i)(?:lob)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}((test|live)_pub_[a-f0-9]{31})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
