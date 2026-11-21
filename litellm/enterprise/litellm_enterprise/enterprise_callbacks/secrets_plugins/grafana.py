"""
This plugin searches for Grafana secrets.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class GrafanaDetector(RegexBasedDetector):
    """Scans for Grafana Secrets."""

    @property
    def secret_type(self) -> str:
        return "Grafana Secret"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Grafana API key or Grafana Cloud API key
            re.compile(
                r"""(?i)\b(eyJrIjoi[A-Za-z0-9]{70,400}={0,2})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Grafana Cloud API token
            re.compile(
                r"""(?i)\b(glc_[A-Za-z0-9+/]{32,400}={0,2})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Grafana Service Account token
            re.compile(
                r"""(?i)\b(glsa_[A-Za-z0-9]{32}_[A-Fa-f0-9]{8})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
