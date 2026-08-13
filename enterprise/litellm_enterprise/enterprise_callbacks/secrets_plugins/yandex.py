import re

from detect_secrets.plugins.base import RegexBasedDetector


class YandexDetector(RegexBasedDetector):
    """Scans for Yandex Access Tokens, API Keys, and AWS Access Tokens."""

    @property
    def secret_type(self) -> str:
        return "Yandex Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Yandex Access Token
            re.compile(
                r"""(?i)(?:yandex)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}(t1\.[A-Z0-9a-z_-]+[=]{0,2}\.[A-Z0-9a-z_-]{86}[=]{0,2})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Yandex API Key
            re.compile(
                r"""(?i)(?:yandex)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}(AQVN[A-Za-z0-9_\-]{35,38})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Yandex AWS Access Token
            re.compile(
                r"""(?i)(?:yandex)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}(YC[a-zA-Z0-9_\-]{38})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
