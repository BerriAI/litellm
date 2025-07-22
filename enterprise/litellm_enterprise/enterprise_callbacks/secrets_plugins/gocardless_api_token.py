"""
This plugin searches for GoCardless API tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class GoCardlessApiTokenDetector(RegexBasedDetector):
    """Scans for GoCardless API Tokens."""

    @property
    def secret_type(self) -> str:
        return "GoCardless API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # GoCardless API token
            re.compile(
                r"""(?:gocardless)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}(live_[a-z0-9\-_=]{40})(?:['|\"|\n|\r|\s|\x60|;]|$)""",
                re.IGNORECASE,
            ),
        ]
