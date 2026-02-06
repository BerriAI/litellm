"""
This plugin searches for Finicity API tokens and Client Secrets.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class FinicityDetector(RegexBasedDetector):
    """Scans for Finicity API tokens and Client Secrets."""

    @property
    def secret_type(self) -> str:
        return "Finicity Credentials"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Finicity API token
            re.compile(
                r"""(?i)(?:finicity)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-f0-9]{32})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Finicity Client Secret
            re.compile(
                r"""(?i)(?:finicity)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{20})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
