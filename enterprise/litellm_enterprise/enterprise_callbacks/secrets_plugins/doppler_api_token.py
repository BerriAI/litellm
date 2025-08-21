"""
This plugin searches for Doppler API tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class DopplerApiTokenDetector(RegexBasedDetector):
    """Scans for Doppler API Tokens."""

    @property
    def secret_type(self) -> str:
        return "Doppler API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Doppler API token
            re.compile(r"""(?i)dp\.pt\.[a-z0-9]{43}"""),
        ]
