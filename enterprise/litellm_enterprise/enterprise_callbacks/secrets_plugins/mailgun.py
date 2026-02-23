"""
This plugin searches for Mailgun API secrets, public validation keys, and webhook signing keys.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class MailgunDetector(RegexBasedDetector):
    """Scans for Mailgun secrets."""

    @property
    def secret_type(self) -> str:
        return "Mailgun Secret"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Mailgun Private API Token
            re.compile(
                r"""(?i)(?:mailgun)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}(key-[a-f0-9]{32})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Mailgun Public Validation Key
            re.compile(
                r"""(?i)(?:mailgun)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}(pubkey-[a-f0-9]{32})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Mailgun Webhook Signing Key
            re.compile(
                r"""(?i)(?:mailgun)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-h0-9]{32}-[a-h0-9]{8}-[a-h0-9]{8})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
