"""Microsoft Teams alert delivery helpers.

Teams incoming webhooks (Workflows and legacy connectors) accept an Adaptive
Card wrapped in a message attachment, so alert text is delivered as a single
wrapped TextBlock.
"""

import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from typing_extensions import ReadOnly, TypedDict

from litellm.types.integrations.slack_alerting import AlertType

MS_TEAMS_WEBHOOK_URL_ENV: Final = "MS_TEAMS_WEBHOOK_URL"

MS_TEAMS_ALERTING_DESTINATION: Final = "ms_teams"

MS_TEAMS_ALERT_HEADERS: Final[Mapping[str, str]] = MappingProxyType({"Content-type": "application/json"})


class MSTeamsTextBlock(TypedDict):
    type: ReadOnly[str]
    text: ReadOnly[str]
    wrap: ReadOnly[bool]


class MSTeamsAdaptiveCard(TypedDict):
    type: ReadOnly[str]
    version: ReadOnly[str]
    body: ReadOnly[tuple[MSTeamsTextBlock, ...]]


class MSTeamsAttachment(TypedDict):
    contentType: ReadOnly[str]
    content: ReadOnly[MSTeamsAdaptiveCard]


class MSTeamsMessage(TypedDict):
    type: ReadOnly[str]
    attachments: ReadOnly[tuple[MSTeamsAttachment, ...]]


class MSTeamsAlertText(TypedDict):
    text: ReadOnly[str]


class MSTeamsQueueItem(TypedDict):
    url: ReadOnly[str]
    headers: ReadOnly[Mapping[str, str]]
    payload: ReadOnly[MSTeamsAlertText]
    alert_type: ReadOnly[AlertType]
    format: ReadOnly[str]


def get_ms_teams_webhook_url() -> str | None:
    return os.getenv(MS_TEAMS_WEBHOOK_URL_ENV)


def build_ms_teams_payload(text: str) -> MSTeamsMessage:
    return MSTeamsMessage(
        type="message",
        attachments=(
            MSTeamsAttachment(
                contentType="application/vnd.microsoft.card.adaptive",
                content=MSTeamsAdaptiveCard(
                    type="AdaptiveCard",
                    version="1.4",
                    body=(MSTeamsTextBlock(type="TextBlock", text=text, wrap=True),),
                ),
            ),
        ),
    )
