"""
This plugin searches for New Relic API tokens and keys.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class NewRelicDetector(RegexBasedDetector):
    """Scans for New Relic API tokens and keys."""

    @property
    def secret_type(self) -> str:
        return "New Relic API Secrets"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # New Relic ingest browser API token
            re.compile(
                r"""(?i)(?:new-relic|newrelic|new_relic)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}(NRJS-[a-f0-9]{19})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # New Relic user API ID
            re.compile(
                r"""(?i)(?:new-relic|newrelic|new_relic)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9]{64})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
            # New Relic user API Key
            re.compile(
                r"""(?i)(?:new-relic|newrelic|new_relic)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}(NRAK-[a-z0-9]{27})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
            ),
        ]
