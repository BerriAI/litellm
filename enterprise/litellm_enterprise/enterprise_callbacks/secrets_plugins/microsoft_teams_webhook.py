"""
This plugin searches for Microsoft Teams Webhook URLs.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class MicrosoftTeamsWebhookDetector(RegexBasedDetector):
    """Scans for Microsoft Teams Webhook URLs."""

    @property
    def secret_type(self) -> str:
        return "Microsoft Teams Webhook"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Microsoft Teams Webhook
            re.compile(
                r"""https:\/\/[a-z0-9]+\.webhook\.office\.com\/webhookb2\/[a-z0-9]{8}-([a-z0-9]{4}-){3}[a-z0-9]{12}@[a-z0-9]{8}-([a-z0-9]{4}-){3}[a-z0-9]{12}\/IncomingWebhook\/[a-z0-9]{32}\/[a-z0-9]{8}-([a-z0-9]{4}-){3}[a-z0-9]{12}"""
            ),
        ]
