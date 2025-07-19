"""
This plugin searches for Age secret keys
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class AgeSecretKeyDetector(RegexBasedDetector):
    """Scans for Age secret keys."""

    @property
    def secret_type(self) -> str:
        return "Age Secret Key"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(r"""AGE-SECRET-KEY-1[QPZRY9X8GF2TVDW0S3JN54KHCE6MUA7L]{58}"""),
        ]
