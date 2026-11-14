import re

from detect_secrets.plugins.base import RegexBasedDetector


class TwitterDetector(RegexBasedDetector):
    """Scans for Twitter Access Secrets, Access Tokens, API Keys, API Secrets, and Bearer Tokens."""

    @property
    def secret_type(self) -> str:
        return "Twitter Secret"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Twitter Access Secret
            re.compile(
                r"""(?i)(?:twitter)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{45})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Twitter Access Token
            re.compile(
                r"""(?i)(?:twitter)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([0-9]{15,25}-[a-zA-Z0-9]{20,40})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Twitter API Key
            re.compile(
                r"""(?i)(?:twitter)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{25})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Twitter API Secret
            re.compile(
                r"""(?i)(?:twitter)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{50})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Twitter Bearer Token
            re.compile(
                r"""(?i)(?:twitter)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}(A{22}[a-zA-Z0-9%]{80,100})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
