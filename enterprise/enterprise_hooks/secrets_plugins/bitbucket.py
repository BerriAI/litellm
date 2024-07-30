"""
This plugin searches for Bitbucket Client ID and Client Secret
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class BitbucketDetector(RegexBasedDetector):
    """Scans for Bitbucket Client ID and Client Secret."""

    @property
    def secret_type(self) -> str:
        return "Bitbucket Secrets"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # For Bitbucket Client ID
            re.compile(
                r"""(?i)(?:bitbucket)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{32})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # For Bitbucket Client Secret
            re.compile(
                r"""(?i)(?:bitbucket)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9=_\-]{64})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
