"""
This plugin searches for Shippo API Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class ShippoApiTokenDetector(RegexBasedDetector):
    """Scans for Shippo API Tokens."""

    @property
    def secret_type(self) -> str:
        return "Shippo API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"""(?i)\b(shippo_(live|test)_[a-f0-9]{40})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            )
        ]
