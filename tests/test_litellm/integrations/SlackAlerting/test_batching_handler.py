import json
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.integrations.SlackAlerting.ms_teams import MS_TEAMS_WEBHOOK_URL_ENV
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.types.integrations.slack_alerting import AlertQueueItem, AlertType

SLACK_WEBHOOK_URL: Final = "https://hooks.slack.com/services/test"
THRESHOLD_ALERT: Final = "User Budget: 15% or less of budget remaining\n\n*user_id:* `user-a`"
CROSSED_ALERT: Final = "User Budget: Budget Crossed\n\n*user_id:* `user-b`"


def _slack_alerting_recording_posts(alerting: list[str]) -> SlackAlerting:
    slack_alerting: Final = SlackAlerting(alerting=alerting)
    slack_alerting.periodic_started = True
    response: Final = MagicMock()
    response.status_code = 200
    slack_alerting.async_http_handler = MagicMock()
    slack_alerting.async_http_handler.post = AsyncMock(return_value=response)
    return slack_alerting


def _queued_slack_alert(text: str) -> AlertQueueItem:
    return {
        "url": SLACK_WEBHOOK_URL,
        "headers": {"Content-type": "application/json"},
        "payload": {"text": text},
        "alert_type": AlertType.budget_alerts,
    }


def _posted_bodies(slack_alerting: SlackAlerting) -> tuple[dict, ...]:
    return tuple(json.loads(call.kwargs["data"]) for call in slack_alerting.async_http_handler.post.call_args_list)


async def _send_budget_alert(slack_alerting: SlackAlerting, message: str) -> None:
    await slack_alerting.send_alert(
        message=message,
        level="High",
        alert_type=AlertType.budget_alerts,
        alerting_metadata={},
    )


@pytest.mark.asyncio
async def test_async_send_batch_delivers_every_distinct_alert_queued_in_one_flush(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", SLACK_WEBHOOK_URL)
    slack_alerting: Final = _slack_alerting_recording_posts(["slack"])
    await _send_budget_alert(slack_alerting, THRESHOLD_ALERT)
    await _send_budget_alert(slack_alerting, CROSSED_ALERT)

    await slack_alerting.async_send_batch()

    posted_texts: Final = tuple(body["text"] for body in _posted_bodies(slack_alerting))
    assert len(posted_texts) == 2
    assert THRESHOLD_ALERT in posted_texts[0]
    assert CROSSED_ALERT in posted_texts[1]
    assert not any(text.startswith("[Num Alerts") for text in posted_texts)
    assert slack_alerting.log_queue == []


@pytest.mark.asyncio
async def test_async_send_batch_collapses_only_identical_alerts():
    slack_alerting: Final = _slack_alerting_recording_posts(["slack"])
    slack_alerting.log_queue.extend(
        (
            _queued_slack_alert(THRESHOLD_ALERT),
            _queued_slack_alert(CROSSED_ALERT),
            _queued_slack_alert(THRESHOLD_ALERT),
        )
    )

    await slack_alerting.async_send_batch()

    assert _posted_bodies(slack_alerting) == (
        {"text": f"[Num Alerts: 2]\n\n{THRESHOLD_ALERT}"},
        {"text": CROSSED_ALERT},
    )


@pytest.mark.asyncio
async def test_async_send_batch_delivers_every_distinct_ms_teams_alert(monkeypatch):
    monkeypatch.setenv(MS_TEAMS_WEBHOOK_URL_ENV, "https://teams.example/webhook")
    slack_alerting: Final = _slack_alerting_recording_posts(["ms_teams"])
    await _send_budget_alert(slack_alerting, THRESHOLD_ALERT)
    await _send_budget_alert(slack_alerting, CROSSED_ALERT)

    await slack_alerting.async_send_batch()

    card_texts: Final = tuple(
        body["attachments"][0]["content"]["body"][0]["text"] for body in _posted_bodies(slack_alerting)
    )
    assert len(card_texts) == 2
    assert THRESHOLD_ALERT in card_texts[0]
    assert CROSSED_ALERT in card_texts[1]
