"""
This plugin searches for Confluent Access Token and Confluent Secret Key
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class ConfluentDetector(RegexBasedDetector):
    """Scans for Confluent Access Token and Confluent Secret Key."""

    @property
    def secret_type(self) -> str:
        return "Confluent Secret"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # For Confluent Access Token
            re.compile(
                r"""(?i)(?:confluent)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{16})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # For Confluent Secret Key
            re.compile(
                r"""(?i)(?:confluent)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{64})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
