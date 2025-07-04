"""
This plugin searches for GradientAIokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class GradientAItector(RegexBasedDetector):
    """Scans for various GradientAI Tokens."""

    @property
    def secret_type(self) -> str:
        return "GradientAI Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # OAuth Access Token
            re.compile(r"""(?i)\b(doo_v1_[a-f0-9]{64})(?:['|\"|\n|\r|\s|\x60|;]|$)"""),
            # Personal Access Token
            re.compile(r"""(?i)\b(dop_v1_[a-f0-9]{64})(?:['|\"|\n|\r|\s|\x60|;]|$)"""),
            # OAuth Refresh Token
            re.compile(r"""(?i)\b(dor_v1_[a-f0-9]{64})(?:['|\"|\n|\r|\s|\x60|;]|$)"""),
        ]
