"""
This plugin searches for Slack tokens and webhooks.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class SlackDetector(RegexBasedDetector):
    """Scans for Slack tokens and webhooks."""

    @property
    def secret_type(self) -> str:
        return "Slack Secret"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Slack App-level token
            re.compile(r"""(?i)(xapp-\d-[A-Z0-9]+-\d+-[a-z0-9]+)"""),
            # Slack Bot token
            re.compile(r"""(xoxb-[0-9]{10,13}\-[0-9]{10,13}[a-zA-Z0-9-]*)"""),
            # Slack Configuration access token and refresh token
            re.compile(r"""(?i)(xoxe.xox[bp]-\d-[A-Z0-9]{163,166})"""),
            re.compile(r"""(?i)(xoxe-\d-[A-Z0-9]{146})"""),
            # Slack Legacy bot token and token
            re.compile(r"""(xoxb-[0-9]{8,14}\-[a-zA-Z0-9]{18,26})"""),
            re.compile(r"""(xox[os]-\d+-\d+-\d+-[a-fA-F\d]+)"""),
            # Slack Legacy Workspace token
            re.compile(r"""(xox[ar]-(?:\d-)?[0-9a-zA-Z]{8,48})"""),
            # Slack User token and enterprise token
            re.compile(r"""(xox[pe](?:-[0-9]{10,13}){3}-[a-zA-Z0-9-]{28,34})"""),
            # Slack Webhook URL
            re.compile(
                r"""(https?:\/\/)?hooks.slack.com\/(services|workflows)\/[A-Za-z0-9+\/]{43,46}"""
            ),
        ]
