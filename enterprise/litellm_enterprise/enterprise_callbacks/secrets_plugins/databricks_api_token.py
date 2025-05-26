"""
This plugin searches for Databricks API token.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class DatabricksApiTokenDetector(RegexBasedDetector):
    """Scans for Databricks API token."""

    @property
    def secret_type(self) -> str:
        return "Databricks API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(r"""(?i)\b(dapi[a-h0-9]{32})(?:['|\"|\n|\r|\s|\x60|;]|$)"""),
        ]
