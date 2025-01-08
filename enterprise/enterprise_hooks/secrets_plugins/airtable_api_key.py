"""
This plugin searches for Airtable API keys
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class AirtableApiKeyDetector(RegexBasedDetector):
    """Scans for Airtable API keys."""

    @property
    def secret_type(self) -> str:
        return "Airtable API Key"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"""(?i)(?:airtable)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{17})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
