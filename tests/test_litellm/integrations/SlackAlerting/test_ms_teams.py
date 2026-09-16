import json
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.SlackAlerting.batching_handler import send_to_webhook
from litellm.integrations.SlackAlerting.ms_teams import (
    MS_TEAMS_ALERTING_DESTINATION,
    MS_TEAMS_WEBHOOK_URL_ENV,
    build_ms_teams_payload,
    get_ms_teams_webhook_url,
)
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.proxy._types import AlertType


def test_build_ms_teams_payload_wraps_text_in_adaptive_card():
    payload: Final = build_ms_teams_payload("hello alert")
    assert payload["type"] == "message"
    attachment: Final = payload["attachments"][0]
    assert attachment["contentType"] == "application/vnd.microsoft.card.adaptive"
    card: Final = attachment["content"]
    assert card["type"] == "AdaptiveCard"
    assert card["body"] == ({"type": "TextBlock", "text": "hello alert", "wrap": True},)


def test_get_ms_teams_webhook_url_reads_env(monkeypatch):
    monkeypatch.setenv(MS_TEAMS_WEBHOOK_URL_ENV, "https://teams.example/webhook")
    assert get_ms_teams_webhook_url() == "https://teams.example/webhook"
    monkeypatch.delenv(MS_TEAMS_WEBHOOK_URL_ENV)
    assert get_ms_teams_webhook_url() is None


@pytest.mark.asyncio
async def test_send_alert_enqueues_ms_teams_item(monkeypatch):
    monkeypatch.setenv(MS_TEAMS_WEBHOOK_URL_ENV, "https://teams.example/webhook")
    slack_alerting: Final = SlackAlerting(alerting=["ms_teams"])
    await slack_alerting.send_alert(
        message="proxy is down",
        level="High",
        alert_type=AlertType.db_exceptions,
        alerting_metadata={},
    )
    assert len(slack_alerting.log_queue) == 1
    item: Final = slack_alerting.log_queue[0]
    assert item["url"] == "https://teams.example/webhook"
    assert item["format"] == MS_TEAMS_ALERTING_DESTINATION
    assert item["alert_type"] == AlertType.db_exceptions
    assert "proxy is down" in item["payload"]["text"]


@pytest.mark.asyncio
async def test_send_alert_ms_teams_missing_webhook_drops_alert(monkeypatch):
    monkeypatch.delenv(MS_TEAMS_WEBHOOK_URL_ENV, raising=False)
    slack_alerting: Final = SlackAlerting(alerting=["ms_teams"])
    await slack_alerting.send_alert(
        message="proxy is down",
        level="High",
        alert_type=AlertType.db_exceptions,
        alerting_metadata={},
    )
    assert len(slack_alerting.log_queue) == 0


@pytest.mark.asyncio
async def test_send_alert_slack_and_ms_teams_enqueue_both(monkeypatch):
    monkeypatch.setenv(MS_TEAMS_WEBHOOK_URL_ENV, "https://teams.example/webhook")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/test")
    slack_alerting: Final = SlackAlerting(alerting=["slack", "ms_teams"])
    await slack_alerting.send_alert(
        message="proxy is down",
        level="High",
        alert_type=AlertType.db_exceptions,
        alerting_metadata={},
    )
    urls: Final = sorted(item["url"] for item in slack_alerting.log_queue)
    assert urls == ["https://hooks.slack.com/services/test", "https://teams.example/webhook"]


@pytest.mark.asyncio
async def test_send_to_webhook_posts_adaptive_card_for_ms_teams_items():
    slack_alerting: Final = SlackAlerting(alerting=["ms_teams"])
    mock_response: Final = MagicMock()
    mock_response.status_code = 200
    slack_alerting.async_http_handler = MagicMock()
    slack_alerting.async_http_handler.post = AsyncMock(return_value=mock_response)

    item: Final = {
        "url": "https://teams.example/webhook",
        "headers": {"Content-type": "application/json"},
        "payload": {"text": "alert body"},
        "alert_type": AlertType.db_exceptions,
        "format": MS_TEAMS_ALERTING_DESTINATION,
    }
    await send_to_webhook(slackAlertingInstance=slack_alerting, item=item, count=1)

    call_kwargs: Final = slack_alerting.async_http_handler.post.call_args.kwargs
    assert call_kwargs["url"] == "https://teams.example/webhook"
    sent_body: Final = json.loads(call_kwargs["data"])
    assert sent_body["type"] == "message"
    assert sent_body["attachments"][0]["content"]["body"][0]["text"] == "alert body"


@pytest.mark.asyncio
async def test_send_to_webhook_keeps_slack_payload_shape():
    slack_alerting: Final = SlackAlerting(alerting=["slack"])
    mock_response: Final = MagicMock()
    mock_response.status_code = 200
    slack_alerting.async_http_handler = MagicMock()
    slack_alerting.async_http_handler.post = AsyncMock(return_value=mock_response)

    item: Final = {
        "url": "https://hooks.slack.com/services/test",
        "headers": {"Content-type": "application/json"},
        "payload": {"text": "alert body"},
        "alert_type": AlertType.db_exceptions,
    }
    await send_to_webhook(slackAlertingInstance=slack_alerting, item=item, count=1)

    call_kwargs: Final = slack_alerting.async_http_handler.post.call_args.kwargs
    assert json.loads(call_kwargs["data"]) == {"text": "alert body"}
