import smtplib
from datetime import datetime, timezone
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import TypeAdapter

from litellm.integrations.performance_alert_notifications import PerformanceAlertNotifier, email_body, slack_text
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.models.alert import AlertSummary, MetricEvidence
from litellm.repositories.alert_repository import AlertRepository
from litellm.repositories.config_repository import ConfigRepository
from litellm.types.integrations.slack_alerting import AlertType

BASE_URL: Final = "https://gateway.example"
WEBHOOK: Final = "https://hooks.slack.example/alerts"
_TEXT: Final = TypeAdapter(dict[str, str])


def summary(index: int) -> AlertSummary:
    return AlertSummary(
        id=str(index),
        type="spend",
        resource_type="team",
        resource_name=f"Research {index} <team>",
        created_at=datetime(2026, 1, 9, tzinfo=timezone.utc),
        evidence=MetricEvidence(value=1200, baseline=200, unit="usd", sample_size=2000),
    )


def test_daily_messages_include_five_findings_and_escape_resource_names() -> None:
    alerts: Final = tuple(summary(index) for index in range(1, 7))
    body, text = email_body(alerts, BASE_URL)
    slack: Final = slack_text(alerts, BASE_URL)
    assert body.count("<li>") == 5
    assert slack.count("?alert=") == 5
    assert text.count("?alert=") == 5
    assert "$1,200.00 vs $200.00" in text
    assert "Research 6" not in body + text + slack
    assert "Research 1 &lt;team&gt;" in body and "Research 1 &lt;team&gt;" in slack
    assert f'<a href="{BASE_URL}/ui/alerts">See all alerts</a>' in body
    assert f"<{BASE_URL}/ui/alerts|See all alerts>" in slack


@pytest.mark.parametrize(
    ("kind", "metric", "expected"),
    (
        ("reliability", MetricEvidence(value=0.25, baseline=0.02, unit="ratio", sample_size=2000), "25.0% vs 2.0%"),
        (
            "latency",
            MetricEvidence(value=4000, baseline=1500, unit="milliseconds", sample_size=2000),
            "4,000 ms vs 1,500 ms",
        ),
        (
            "traffic",
            MetricEvidence(value=12000, baseline=2000, unit="requests", sample_size=2000),
            "12,000 requests vs 2,000 requests",
        ),
    ),
)
def test_daily_messages_preserve_metric_units(
    kind: Literal["reliability", "latency", "traffic"], metric: MetricEvidence, expected: str
) -> None:
    alert: Final = summary(1).model_copy(update={"type": kind, "evidence": metric})
    body, text = email_body((alert,), BASE_URL)
    assert expected in body and expected in text
    assert expected in slack_text((alert,), BASE_URL)


@pytest.mark.asyncio
async def test_connected_channels_do_not_subscribe_to_performance_summaries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", BASE_URL)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_SENDER_EMAIL", "alerts@example.com")
    client: Final = MagicMock()
    client.writer_db.litellm_config.find_unique = AsyncMock(return_value=None)
    client.writer_db.tx = MagicMock()
    slack: Final = SlackAlerting(alerting=["slack"], default_webhook_url=WEBHOOK)
    notifier: Final = PerformanceAlertNotifier(ConfigRepository(client, use_writer=True), AlertRepository(client))
    await notifier.run(slack)
    assert slack.performance_alert_webhooks() == ()
    client.writer_db.tx.assert_not_called()


@pytest.mark.asyncio
async def test_slack_summary_reuses_the_configured_connection_and_reports_delivery() -> None:
    posted: Final = MagicMock(return_value=httpx.Response(200, text="ok"))
    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(posted))
    try:
        slack: Final = SlackAlerting(
            alerting=["slack"],
            alert_types=[AlertType.performance_alerts],
            default_webhook_url=WEBHOOK,
            async_http_handler=handler,
        )
        assert slack.performance_alert_webhooks() == (WEBHOOK,)
        message: Final = slack_text((summary(1),), BASE_URL)
        assert await slack.send_performance_alert_summary(WEBHOOK, message)
        request: Final = posted.call_args.args[0]
        assert str(request.url) == WEBHOOK
        assert _TEXT.validate_json(request.content) == {"text": message}
    finally:
        await handler.close()


@pytest.mark.asyncio
async def test_opted_in_summary_reaches_slack_from_the_notification_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", BASE_URL)
    monkeypatch.delenv("SMTP_HOST", raising=False)
    alert: Final = summary(1)
    client: Final = MagicMock()
    client.writer_db.litellm_config.find_unique = AsyncMock(return_value=None)
    client.writer_db.query_raw = AsyncMock(return_value=[{"param_name": "receipt"}])
    client.writer_db.tx.return_value.__aenter__.return_value.query_raw = AsyncMock(
        side_effect=((), ({"id": 1},), (), (alert.model_dump(mode="json"),))
    )
    posted: Final = MagicMock(return_value=httpx.Response(200, text="ok"))
    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(posted))
    try:
        slack: Final = SlackAlerting(
            alerting=["slack"],
            alert_types=[AlertType.performance_alerts],
            default_webhook_url=WEBHOOK,
            async_http_handler=handler,
        )
        notifier: Final = PerformanceAlertNotifier(ConfigRepository(client, use_writer=True), AlertRepository(client))
        await notifier.run(slack, now=alert.created_at)
        posted.assert_called_once()
        assert _TEXT.validate_json(posted.call_args.args[0].content) == {"text": slack_text((alert,), BASE_URL)}
    finally:
        await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (httpx.ReadTimeout("Webhook timed out"), httpx.Response(503)))
async def test_unavailable_webhook_does_not_block_other_destinations(
    monkeypatch: pytest.MonkeyPatch, failure: Exception | httpx.Response
) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", BASE_URL)
    monkeypatch.delenv("SMTP_HOST", raising=False)
    alert: Final = summary(1)
    client: Final = MagicMock()
    client.writer_db.litellm_config.find_unique = AsyncMock(return_value=None)
    client.writer_db.query_raw = AsyncMock(return_value=[{"param_name": "receipt"}])
    client.writer_db.tx.return_value.__aenter__.return_value.query_raw = AsyncMock(
        side_effect=((), ({"id": 1},), (), (alert.model_dump(mode="json"),)) * 2
    )
    posted: Final = MagicMock(side_effect=(failure, httpx.Response(200)))
    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(posted))
    try:
        slack: Final = SlackAlerting(
            alerting=["slack"],
            alert_types=[AlertType.performance_alerts],
            alert_to_webhook_url={AlertType.performance_alerts: [WEBHOOK, WEBHOOK + "/second", WEBHOOK]},
            async_http_handler=handler,
        )
        notifier: Final = PerformanceAlertNotifier(ConfigRepository(client, use_writer=True), AlertRepository(client))
        await notifier.run(slack, now=alert.created_at)
        assert tuple(str(call.args[0].url) for call in posted.call_args_list) == (WEBHOOK, WEBHOOK + "/second")
        assert _TEXT.validate_json(posted.call_args.args[0].content) == {"text": slack_text((alert,), BASE_URL)}
    finally:
        await handler.close()


@pytest.mark.asyncio
async def test_opted_in_email_reaches_recipients_across_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", BASE_URL)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "25")
    monkeypatch.setenv("SMTP_SENDER_EMAIL", "alerts@example.com")
    monkeypatch.setenv("SMTP_TLS", "False")
    monkeypatch.setenv("SMTP_USE_SSL", "False")
    monkeypatch.delenv("SMTP_USERNAME", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    smtp: Final = MagicMock()
    monkeypatch.setattr(smtplib, "SMTP", MagicMock(return_value=smtp))
    alert: Final = summary(1)
    client: Final = MagicMock()
    client.writer_db.litellm_config.find_unique = AsyncMock(
        side_effect=(MagicMock(param_value={"email_settings": {"Performance Alerts": True}}), None, None)
    )
    client.writer_db.query_raw = AsyncMock(return_value=[{"param_name": "receipt"}])
    client.writer_db.tx.return_value.__aenter__.return_value.query_raw = AsyncMock(
        side_effect=(
            (),
            ({"user_id": "a", "email": "first@example.com"},),
            (),
            ({"id": 1},),
            (),
            (alert.model_dump(mode="json"),),
            (),
            ({"user_id": "b", "email": "second@example.com"},),
            (),
            ({"id": 1},),
            (),
            (alert.model_dump(mode="json"),),
            (),
            (),
        )
    )
    notifier: Final = PerformanceAlertNotifier(ConfigRepository(client, use_writer=True), AlertRepository(client))
    await notifier.run(None, now=alert.created_at)
    sends: Final = smtp.__enter__.return_value.send_message.call_args_list
    assert tuple(call.kwargs["to_addrs"] for call in sends) == ("first@example.com", "second@example.com")
    for call in sends:
        message: Final = call.kwargs["msg"]
        assert message["Subject"] == "New performance alerts"
        assert f"{BASE_URL}/ui/alerts?alert=1" in message.get_payload(0).get_payload(decode=True).decode()
        assert message["Message-ID"]
    assert sends[0].kwargs["msg"]["Message-ID"] != sends[1].kwargs["msg"]["Message-ID"]
