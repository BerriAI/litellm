"""
This plugin searches for Telegram Bot API Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class TelegramBotApiTokenDetector(RegexBasedDetector):
    """Scans for Telegram Bot API Tokens."""

    @property
    def secret_type(self) -> str:
        return "Telegram Bot API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"""(?i)(?:^|[^0-9])([0-9]{5,16}:A[a-zA-Z0-9_\-]{34})(?:$|[^a-zA-Z0-9_\-])"""
            )
        ]
