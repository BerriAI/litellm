"""
This plugin searches for Pulumi API Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class PulumiApiTokenDetector(RegexBasedDetector):
    """Scans for Pulumi API Tokens."""

    @property
    def secret_type(self) -> str:
        return "Pulumi API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [re.compile(r"""(?i)\b(pul-[a-f0-9]{40})(?:['|\"|\n|\r|\s|\x60|;]|$)""")]
