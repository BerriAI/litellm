from __future__ import annotations

import asyncio
import hashlib
import html
import os
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final

from pydantic import AliasPath, BaseModel, ConfigDict, Field

from litellm._logging import verbose_proxy_logger
from litellm.models.alert import AlertSummary
from litellm.repositories.alert_repository import AlertRepository
from litellm.repositories.config_repository import ConfigRepository

if TYPE_CHECKING:
    from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting


class _EmailSettings(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    enabled: bool = Field(False, strict=True, validation_alias=AliasPath("email_settings", "Performance Alerts"))


class _Delivery(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    revision: int = 1
    cursor: int
    first_window_start: datetime
    next_send_at: datetime
    lease_until: datetime


def _description(alert: AlertSummary) -> str:
    labels: Final = {
        "spend": "Spend spike",
        "traffic": "Traffic spike",
        "latency": "Slower responses",
        "reliability": "More failed requests",
    }
    metric: Final = alert.evidence

    def amount(value: float) -> str:
        match metric.unit:
            case "usd":
                return f"${value:,.2f}"
            case "ratio":
                return f"{value:.1%}"
            case "milliseconds":
                return f"{value:,.0f} ms"
            case "requests":
                return f"{value:,.0f} requests"

    return f"{labels[alert.type]} for {alert.resource_name}: {amount(metric.value)} vs {amount(metric.baseline)}"


def slack_text(alerts: Sequence[AlertSummary], base_url: str) -> str:
    rows: Final = tuple(
        f"<{base_url}/ui/alerts?alert={alert.id}|{html.escape(_description(alert), quote=False)}>"
        for alert in alerts[:5]
    )
    return "New performance alerts\n\n" + "\n".join(rows) + f"\n\n<{base_url}/ui/alerts|See all alerts>"


def email_body(alerts: Sequence[AlertSummary], base_url: str) -> tuple[str, str]:
    rows: Final = tuple(
        f'<li><a href="{html.escape(base_url, quote=True)}/ui/alerts?alert={alert.id}">{html.escape(_description(alert))}</a></li>'
        for alert in alerts[:5]
    )
    text: Final = (
        "New performance alerts\n\n"
        + "\n".join(f"{_description(alert)}\n{base_url}/ui/alerts?alert={alert.id}" for alert in alerts[:5])
        + f"\n\nSee all alerts: {base_url}/ui/alerts"
    )
    return (
        f'<h2>New performance alerts</h2><ul>{"".join(rows)}</ul><p><a href="{html.escape(base_url, quote=True)}/ui/alerts">See all alerts</a></p>',
        text,
    )


class PerformanceAlertNotifier:
    def __init__(self, config: ConfigRepository, alerts: AlertRepository) -> None:
        self.config: Final = config
        self.alerts: Final = alerts

    async def deliver(
        self,
        destination: str,
        send: Callable[[Sequence[AlertSummary], str], Awaitable[bool]],
        now: datetime,
    ) -> None:
        key: Final = "performance_alert_digest:" + hashlib.sha256(destination.encode()).hexdigest()
        row: Final = await self.config.get_param(key)
        previous: Final = (
            _Delivery.model_validate(row.param_value)
            if row
            else _Delivery(
                revision=0,
                cursor=0,
                first_window_start=now - timedelta(days=1),
                next_send_at=now,
                lease_until=now,
            )
        )
        if previous.next_send_at > now or previous.lease_until > now:
            return
        through: Final = await self.alerts.latest_id()
        found: Final = await self.alerts.summaries(previous.cursor, through)
        alerts: Final = (
            found
            if previous.cursor
            else tuple(alert for alert in found if alert.created_at >= previous.first_window_start)
        )
        claim: Final = previous.model_copy(
            update={
                "revision": previous.revision + 1,
                "lease_until": now + timedelta(minutes=2),
                "next_send_at": now + timedelta(days=1) if alerts else now,
            }
        )
        if not await self.config.set_param_if_revision(key, claim.model_dump(mode="json"), previous.revision):
            return
        if alerts and not await send(alerts, f"{key}-{through}"):
            return
        sent: Final = claim.model_copy(
            update={
                "revision": claim.revision + 1,
                "cursor": through,
                "next_send_at": now + timedelta(days=1) if alerts else now,
                "lease_until": now,
            }
        )
        if not await self.config.set_param_if_revision(key, sent.model_dump(mode="json"), claim.revision):
            verbose_proxy_logger.warning("Could not record performance alert summary delivery")

    async def run(self, slack: SlackAlerting | None, *, now: datetime | None = None) -> None:
        from litellm.proxy.utils import send_email

        base_url: Final = os.getenv("PROXY_BASE_URL", "").rstrip("/")
        if not base_url:
            return
        if slack is not None:
            for webhook in slack.performance_alert_webhooks():

                async def send_slack(alerts: Sequence[AlertSummary], receipt: str, webhook: str = webhook) -> bool:
                    return await slack.send_performance_alert_summary(webhook, slack_text(alerts, base_url))

                await self._try_deliver("slack:" + webhook, send_slack, now or datetime.now(timezone.utc))
        if not os.getenv("SMTP_HOST") or not os.getenv("SMTP_SENDER_EMAIL"):
            return
        settings: Final = await self.config.get_param("general_settings")
        if not _EmailSettings.model_validate(settings.param_value if settings else {}).enabled:
            return
        after: str = ""  # rebind-ok: advances the bounded recipient page
        while recipients := await self.alerts.recipients(after):
            for recipient in recipients:

                async def send_mail(
                    alerts: Sequence[AlertSummary], receipt: str, address: str = recipient.email
                ) -> bool:
                    body, text = email_body(alerts, base_url)
                    return await send_email(
                        receiver_email=address,
                        subject="New performance alerts",
                        html=body,
                        text=text,
                        message_id=f"<{receipt}@alerts.litellm.ai>",
                    )

                await self._try_deliver("email:" + recipient.email, send_mail, now or datetime.now(timezone.utc))
            after = recipients[-1].user_id
            await asyncio.sleep(0.1)

    async def _try_deliver(
        self, destination: str, send: Callable[[Sequence[AlertSummary], str], Awaitable[bool]], now: datetime
    ) -> None:
        try:
            await asyncio.wait_for(self.deliver(destination, send, now), timeout=60)
        except Exception as exc:  # noqa: BLE001  # one unavailable destination must not stop the other recipients
            verbose_proxy_logger.warning("Performance alert summary delivery failed (%s)", type(exc).__name__)
