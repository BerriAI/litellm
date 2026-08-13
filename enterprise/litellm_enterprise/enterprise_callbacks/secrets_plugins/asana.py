"""
This plugin searches for Asana secrets
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class AsanaSecretDetector(RegexBasedDetector):
    """Scans for Asana Client IDs and Client Secrets."""

    @property
    def secret_type(self) -> str:
        return "Asana Secrets"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # For Asana Client ID
            re.compile(
                r"""(?i)(?:asana)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([0-9]{16})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # For Asana Client Secret
            re.compile(
                r"""(?i)(?:asana)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{32})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
