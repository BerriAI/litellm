"""
This plugin searches for Adobe keys
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class AdobeSecretDetector(RegexBasedDetector):
    """Scans for Adobe client keys."""

    @property
    def secret_type(self) -> str:
        return "Adobe Client Keys"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Adobe Client ID (OAuth Web)
            re.compile(
                r"""(?i)(?:adobe)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-f0-9]{32})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Adobe Client Secret
            re.compile(r"(?i)\b((p8e-)[a-z0-9]{32})(?:['|\"|\n|\r|\s|\x60|;]|$)"),
        ]
