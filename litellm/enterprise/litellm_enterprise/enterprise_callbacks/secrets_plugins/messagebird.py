"""
This plugin searches for MessageBird API tokens and client IDs.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class MessageBirdDetector(RegexBasedDetector):
    """Scans for MessageBird secrets."""

    @property
    def secret_type(self) -> str:
        return "MessageBird Secret"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # MessageBird API Token
            re.compile(
                r"""(?i)(?:messagebird|message-bird|message_bird)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{25})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # MessageBird Client ID
            re.compile(
                r"""(?i)(?:messagebird|message-bird|message_bird)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
