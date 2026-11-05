"""Deliver persisted usage incidents through the existing configured transports."""

import json
import os
from html import escape
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

from pydantic import TypeAdapter

from litellm.types.integrations.slack_alerting import AlertType
from litellm.types.proxy.usage_alerts import UsageAlert, UsageAlertChannel, UsageAlertRule

if TYPE_CHECKING:
    from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting

_URLS: Final = TypeAdapter(tuple[str, ...])


def configured_destinations(slack: "SlackAlerting") -> tuple[str, ...]:
    return _URLS.validate_python(slack.alerting or ())  # pyright: ignore[reportUnknownMemberType]  # validate legacy destination config


def activity_url(alert: UsageAlert, base_url: str | None) -> str:
    if not base_url:
        return alert.activity_path
    parsed: Final = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return alert.activity_path
    return base_url.rstrip("/") + "/ui" + alert.activity_path


def incident_message(alert: UsageAlert, url: str | None) -> str:
    scope: Final = "Gateway" if alert.scope_type == "deployment" else f"{alert.scope_type}: {alert.scope_id}"
    unit: Final = " ms" if alert.metric == "latency" else " USD" if alert.metric == "spend" else ""
    return (
        f"{alert.rule_name}\n{scope}\n{alert.day.isoformat()} UTC: "
        f"{alert.observed_value:,.2f}{unit}; baseline {alert.baseline_value:,.2f}{unit}; "
        f"threshold {alert.threshold:,.2f}{unit}." + (f"\nView activity: {url}" if url is not None else "")
    )


class UsageAlertDelivery:
    def __init__(self, slack: "SlackAlerting") -> None:
        self._slack = slack

    async def send(self, alert: UsageAlert, rule: UsageAlertRule, channel: UsageAlertChannel) -> bool:
        configured: Final = configured_destinations(self._slack)
        if channel not in configured:
            return False
        url: Final = activity_url(alert, os.getenv("PROXY_BASE_URL"))
        message: Final = incident_message(alert, None if channel == "email" else url)
        if channel == "email":
            from litellm.proxy.utils import send_email

            for recipient in rule.email_recipients:
                await send_email(
                    receiver_email=recipient,
                    subject=f"LiteLLM usage alert: {alert.rule_name}",
                    html=(
                        f"<p>{escape(message).replace(chr(10), '<br>')}</p>"
                        f'<p><a href="{escape(url, quote=True)}">View activity</a></p>'
                    ),
                    raise_on_error=True,
                )
            return bool(rule.email_recipients)
        overrides: Final = self._slack.alert_to_webhook_url or {}
        configured_url: Final = (
            overrides.get(AlertType.usage_alerts)
            or self._slack.default_webhook_url
            or os.getenv("SLACK_WEBHOOK_URL")
            or os.getenv("ALERTING_WEBHOOK_URL")
        )
        if not configured_url:
            return False
        urls: Final = (configured_url,) if isinstance(configured_url, str) else _URLS.validate_python(configured_url)
        for webhook in urls:
            await self._post(webhook, message)
        return True

    async def _post(self, webhook: str, message: str) -> None:
        response: Final = await self._slack.async_http_handler.post(  # pyright: ignore[reportUnknownMemberType]  # legacy HTTP signature; supplied arguments are concrete
            url=webhook,
            headers={"Content-type": "application/json"},
            data=json.dumps({"text": message}),
            timeout=10,
        )
        response.raise_for_status()
