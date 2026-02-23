"""
This plugin searches for SendGrid API Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class SendGridApiTokenDetector(RegexBasedDetector):
    """Scans for SendGrid API Tokens."""

    @property
    def secret_type(self) -> str:
        return "SendGrid API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"""(?i)\b(SG\.[a-z0-9=_\-\.]{66})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            )
        ]
