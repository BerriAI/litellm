"""
This plugin searches for SendinBlue API Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class SendinBlueApiTokenDetector(RegexBasedDetector):
    """Scans for SendinBlue API Tokens."""

    @property
    def secret_type(self) -> str:
        return "SendinBlue API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"""(?i)\b(xkeysib-[a-f0-9]{64}-[a-z0-9]{16})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            )
        ]
