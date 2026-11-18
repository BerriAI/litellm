import json
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import TypeAdapter

from litellm.integrations.SlackAlerting.batching_handler import send_to_webhook
from litellm.integrations.SlackAlerting.ms_teams import (
    MS_TEAMS_ALERTING_DESTINATION,
    MS_TEAMS_WEBHOOK_URL_ENV,
    MSTeamsMessage,
    build_ms_teams_payload,
    get_ms_teams_webhook_url,
)
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import AlertType

_MS_TEAMS_MESSAGE: Final = TypeAdapter(MSTeamsMessage)


def _webhook_accepting_posts() -> AsyncMock:
    response: Final = MagicMock(spec=httpx.Response)
    response.status_code = 200
    http_handler: Final = AsyncMock(spec=AsyncHTTPHandler)
    http_handler.post.return_value = response
    return http_handler


def _posted_card_texts(http_handler: AsyncMock) -> tuple[str, ...]:
    return tuple(
        _MS_TEAMS_MESSAGE.validate_json(call.kwargs["data"])["attachments"][0]["content"]["body"][0]["text"]
        for call in http_handler.post.call_args_list
    )


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
    http_handler: Final = _webhook_accepting_posts()
    slack_alerting: Final = SlackAlerting(alerting=["ms_teams"], async_http_handler=http_handler)

    item: Final = {
        "url": "https://teams.example/webhook",
        "headers": {"Content-type": "application/json"},
        "payload": {"text": "alert body"},
        "alert_type": AlertType.db_exceptions,
        "format": MS_TEAMS_ALERTING_DESTINATION,
    }
    await send_to_webhook(slackAlertingInstance=slack_alerting, item=item, count=1)

    call_kwargs: Final = http_handler.post.call_args.kwargs
    assert call_kwargs["url"] == "https://teams.example/webhook"
    sent_body: Final = json.loads(call_kwargs["data"])
    assert sent_body["type"] == "message"
    assert sent_body["attachments"][0]["content"]["body"][0]["text"] == "alert body"


@pytest.mark.asyncio
async def test_send_to_webhook_keeps_slack_payload_shape():
    http_handler: Final = _webhook_accepting_posts()
    slack_alerting: Final = SlackAlerting(alerting=["slack"], async_http_handler=http_handler)

    item: Final = {
        "url": "https://hooks.slack.com/services/test",
        "headers": {"Content-type": "application/json"},
        "payload": {"text": "alert body"},
        "alert_type": AlertType.db_exceptions,
    }
    await send_to_webhook(slackAlertingInstance=slack_alerting, item=item, count=1)

    call_kwargs: Final = http_handler.post.call_args.kwargs
    assert json.loads(call_kwargs["data"]) == {"text": "alert body"}


@pytest.mark.asyncio
async def test_async_send_batch_delivers_every_distinct_ms_teams_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MS_TEAMS_WEBHOOK_URL_ENV, "https://teams.example/webhook")
    http_handler: Final = _webhook_accepting_posts()
    slack_alerting: Final = SlackAlerting(alerting=["ms_teams"], async_http_handler=http_handler)
    slack_alerting.periodic_started = True
    for message in ("User Budget: 15% or less of budget remaining", "User Budget: Budget Crossed"):
        await slack_alerting.send_alert(
            message=message,
            level="High",
            alert_type=AlertType.budget_alerts,
            alerting_metadata={},
        )

    await slack_alerting.async_send_batch()

    card_texts: Final = _posted_card_texts(http_handler)
    assert len(card_texts) == 2
    assert "User Budget: 15% or less of budget remaining" in card_texts[0]
    assert "User Budget: Budget Crossed" in card_texts[1]
