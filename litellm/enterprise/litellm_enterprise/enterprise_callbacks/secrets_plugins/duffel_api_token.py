"""
This plugin searches for Duffel API Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class DuffelApiTokenDetector(RegexBasedDetector):
    """Scans for Duffel API Tokens."""

    @property
    def secret_type(self) -> str:
        return "Duffel API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Duffel API Token
            re.compile(r"""(?i)duffel_(test|live)_[a-z0-9_\-=]{43}"""),
        ]
