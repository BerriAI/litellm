"""
This plugin searches for Clojars API tokens
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class ClojarsApiTokenDetector(RegexBasedDetector):
    """Scans for Clojars API tokens."""

    @property
    def secret_type(self) -> str:
        return "Clojars API token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # For Clojars API token
            re.compile(r"(?i)(CLOJARS_)[a-z0-9]{60}"),
        ]
