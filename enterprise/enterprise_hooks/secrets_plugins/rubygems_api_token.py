"""
This plugin searches for Rubygem API Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class RubygemsApiTokenDetector(RegexBasedDetector):
    """Scans for Rubygem API Tokens."""

    @property
    def secret_type(self) -> str:
        return "Rubygem API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(r"""(?i)\b(rubygems_[a-f0-9]{48})(?:['|\"|\n|\r|\s|\x60|;]|$)""")
        ]
