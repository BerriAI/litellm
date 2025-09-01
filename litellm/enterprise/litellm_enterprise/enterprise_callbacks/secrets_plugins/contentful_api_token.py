"""
This plugin searches for Contentful delivery API token.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class ContentfulApiTokenDetector(RegexBasedDetector):
    """Scans for Contentful delivery API token."""

    @property
    def secret_type(self) -> str:
        return "Contentful API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"""(?i)(?:contentful)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9=_\-]{43})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
