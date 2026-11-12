from datetime import datetime, timezone
from typing import Final
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
