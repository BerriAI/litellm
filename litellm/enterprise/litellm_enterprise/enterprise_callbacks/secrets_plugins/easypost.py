"""
This plugin searches for EasyPost tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class EasyPostDetector(RegexBasedDetector):
    """Scans for various EasyPost Tokens."""

    @property
    def secret_type(self) -> str:
        return "EasyPost Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # EasyPost API token
            re.compile(r"""(?i)\bEZAK[a-z0-9]{54}"""),
            # EasyPost test API token
            re.compile(r"""(?i)\bEZTK[a-z0-9]{54}"""),
        ]
