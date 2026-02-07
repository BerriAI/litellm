import re

from detect_secrets.plugins.base import RegexBasedDetector


class SumoLogicDetector(RegexBasedDetector):
    """Scans for SumoLogic Access ID and Access Token."""

    @property
    def secret_type(self) -> str:
        return "SumoLogic"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"""(?i:(?:sumo)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3})(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}(su[a-zA-Z0-9]{12})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            re.compile(
                r"""(?i)(?:sumo)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{64})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
