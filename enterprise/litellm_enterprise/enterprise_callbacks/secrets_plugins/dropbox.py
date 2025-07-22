"""
This plugin searches for Dropbox tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class DropboxDetector(RegexBasedDetector):
    """Scans for various Dropbox Tokens."""

    @property
    def secret_type(self) -> str:
        return "Dropbox Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Dropbox API secret
            re.compile(
                r"""(?i)(?:dropbox)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{15})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Dropbox long-lived API token
            re.compile(
                r"""(?i)(?:dropbox)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{11}(AAAAAAAAAA)[a-z0-9\-_=]{43})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Dropbox short-lived API token
            re.compile(
                r"""(?i)(?:dropbox)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}(sl\.[a-z0-9\-=_]{135})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
