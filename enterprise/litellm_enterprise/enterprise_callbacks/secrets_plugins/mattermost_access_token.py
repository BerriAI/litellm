"""
This plugin searches for Mattermost Access Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class MattermostAccessTokenDetector(RegexBasedDetector):
    """Scans for Mattermost Access Tokens."""

    @property
    def secret_type(self) -> str:
        return "Mattermost Access Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Mattermost Access Token
            re.compile(
                r"""(?i)(?:mattermost)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{26})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
