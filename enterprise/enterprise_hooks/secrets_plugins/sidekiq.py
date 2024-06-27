"""
This plugin searches for Sidekiq secrets and sensitive URLs.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class SidekiqDetector(RegexBasedDetector):
    """Scans for Sidekiq secrets and sensitive URLs."""

    @property
    def secret_type(self) -> str:
        return "Sidekiq Secret"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Sidekiq Secret
            re.compile(
                r"""(?i)(?:BUNDLE_ENTERPRISE__CONTRIBSYS__COM|BUNDLE_GEMS__CONTRIBSYS__COM)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-f0-9]{8}:[a-f0-9]{8})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # Sidekiq Sensitive URL
            re.compile(
                r"""(?i)\b(http(?:s??):\/\/)([a-f0-9]{8}:[a-f0-9]{8})@(?:gems.contribsys.com|enterprise.contribsys.com)(?:[\/|\#|\?|:]|$)"""
            ),
        ]
