import asyncio
import smtplib
import threading
from contextlib import suppress
from datetime import date, datetime, timezone
from typing import Final

import pytest

from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.integrations.SlackAlerting.usage_alerts import UsageAlertDelivery
from litellm.types.proxy.usage_alerts import UsageAlert, UsageAlertRule

_NOW: Final = datetime(2026, 10, 20, tzinfo=timezone.utc)
_RULE: Final = UsageAlertRule(
    name="Request spike",
    id="rule",
    scope_type="key",
    scope_id="reporting-key",
    metric="requests",
    direction="increase",
    created_at=_NOW,
    updated_at=_NOW,
    created_by=None,
    notification_channels=("slack",),
)
_ALERT: Final = UsageAlert(
    id="incident",
    rule_id="rule",
    rule_name="Request spike",
    scope_type="key",
    scope_id="reporting-key",
    metric="requests",
    direction="increase",
    status="open",
    day=date(2026, 10, 20),
    observed_value=1000,
    baseline_value=200,
    threshold=600,
    first_seen=_NOW,
    last_seen=_NOW,
    baseline_start=date(2026, 10, 13),
    baseline_end=date(2026, 10, 19),
    baseline_sample_days=7,
    activity_path="/api-keys?key=reporting-key&detail_tab=activity",
)


@pytest.mark.asyncio
async def test_cancelled_smtp_attempt_retains_only_one_send(monkeypatch: pytest.MonkeyPatch) -> None:
    connected: Final = asyncio.Event()
    release: Final = asyncio.Event()
    accepted: Final[list[asyncio.StreamWriter]] = []

    async def stalled(_: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        accepted.append(writer)
        connected.set()
        await release.wait()
        writer.close()
        await writer.wait_closed()

    server: Final = await asyncio.start_server(stalled, "127.0.0.1", 0)
    monkeypatch.setenv("SMTP_HOST", "127.0.0.1")
    monkeypatch.setenv("SMTP_PORT", str(server.sockets[0].getsockname()[1]))
    monkeypatch.setenv("SMTP_SENDER_EMAIL", "alerts@example.com")
    monkeypatch.setenv("SMTP_TLS", "False")
    monkeypatch.setenv("SMTP_USE_SSL", "False")
    monkeypatch.delenv("SMTP_USERNAME", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    slack: Final = SlackAlerting(alerting=["email"], alert_types=[])
    delivery: Final = UsageAlertDelivery(slack)
    rule: Final = _RULE.model_copy(
        update={"notification_channels": ("email",), "email_recipients": ("qa@example.com",)}
    )
    try:
        sending = asyncio.create_task(delivery.send(_ALERT, rule, "email"))
        await asyncio.wait_for(connected.wait(), timeout=10)
        sending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sending
        threads: Final = threading.active_count()
        pending: Final = UsageAlertDelivery._email_task
        assert pending is not None and not pending.done()
        for _ in range(100):
            with pytest.raises(RuntimeError, match="still in flight"):
                await delivery.send(_ALERT, rule, "email")
            assert UsageAlertDelivery._email_task is pending
        assert len(accepted) == 1 and threading.active_count() == threads
        with pytest.raises(smtplib.SMTPServerDisconnected, match="timed out"):
            await asyncio.wait_for(asyncio.shield(pending), timeout=10)
        assert pending.done()
    finally:
        release.set()
        server.close()
        await server.wait_closed()
        for writer in accepted:
            writer.close()
        pending = UsageAlertDelivery._email_task
        if pending is not None:
            with suppress(smtplib.SMTPServerDisconnected):
                await asyncio.wait_for(asyncio.shield(pending), timeout=10)
        UsageAlertDelivery._email_task = None
        UsageAlertDelivery._email_key = None
