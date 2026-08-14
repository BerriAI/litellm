"""
This plugin searches for Alibaba secrets
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class AlibabaSecretDetector(RegexBasedDetector):
    """Scans for Alibaba AccessKey IDs and Secret Keys."""

    @property
    def secret_type(self) -> str:
        return "Alibaba Secrets"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # For Alibaba AccessKey ID
            re.compile(r"""(?i)\b((LTAI)[a-z0-9]{20})(?:['|\"|\n|\r|\s|\x60|;]|$)"""),
            # For Alibaba Secret Key
            re.compile(
                r"""(?i)(?:alibaba)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{30})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
