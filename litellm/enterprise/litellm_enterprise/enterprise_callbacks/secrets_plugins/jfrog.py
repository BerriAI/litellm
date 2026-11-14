"""
This plugin searches for JFrog-related secrets like API Key and Identity Token.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class JFrogDetector(RegexBasedDetector):
    """Scans for JFrog-related secrets."""

    @property
    def secret_type(self) -> str:
        return "JFrog Secrets"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # JFrog API Key
            re.compile(
                r"""(?i)(?:jfrog|artifactory|bintray|xray)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{73})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # JFrog Identity Token
            re.compile(
                r"""(?i)(?:jfrog|artifactory|bintray|xray)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{64})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
