"""Deliver persisted usage incidents through the existing configured transports."""

import asyncio
import hashlib
import json
import os
from html import escape
from itertools import islice
from typing import TYPE_CHECKING, ClassVar, Final
from urllib.parse import urlencode, urlsplit
from uuid import NAMESPACE_URL, uuid5

from httpx import Response
from pydantic import TypeAdapter

from litellm.models.issues import IssueStore
from litellm.types.integrations.slack_alerting import AlertType
from litellm.types.proxy.issues import IssueDeliveryBinding, IssueDeliveryInput, IssuePendingDelivery
from litellm.types.proxy.usage_alerts import UsageAlert

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
    unit: Final = (
        " ms"
        if alert.metric == "latency"
        else " USD"
        if alert.metric == "spend"
        else "%"
        if alert.metric == "error_rate"
        else ""
    )
    return (
        f"{alert.rule_name}\n{scope}\n{alert.day.isoformat()} UTC: "
        f"{alert.observed_value:,.2f}{unit}; baseline {alert.baseline_value:,.2f}{unit}; "
        f"threshold {alert.threshold:,.2f}{unit}." + (f"\nView activity: {url}" if url is not None else "")
    )


class UsageAlertDelivery:
    # One retained send per worker bounds SMTP threads even when a job is cancelled.
    _email_task: ClassVar[asyncio.Task[None] | None] = None
    _email_key: ClassVar[str | None] = None
    _email_store: ClassVar[IssueStore | None] = None

    def __init__(self, slack: "SlackAlerting", store: IssueStore | None = None) -> None:
        self._slack = slack
        self._store = store

    async def send_issue(self, pending: IssuePendingDelivery) -> bool:
        urls: Final = self._webhooks() if pending.delivery.channel == "slack" else ()
        if len(urls) > 1 and pending.delivery.slack_destination is None:
            if self._store is None:
                raise RuntimeError("Multiple alert destinations require a delivery store")
            bindings: Final = (
                IssueDeliveryBinding(
                    delivery=IssueDeliveryInput(
                        id=str(uuid5(NAMESPACE_URL, f"{pending.delivery.id}:{destination}")),
                        issue_id=pending.issue.id,
                        channel="slack",
                        slack_destination=destination,
                        parent_delivery_id=pending.delivery.id,
                    ),
                    rule_id=pending.rule.id,
                    rule_revision=pending.rule.updated_at.isoformat(),
                    source_id=pending.alert.id,
                )
                for destination in (hashlib.sha256(url.encode()).hexdigest() for url in urls)
            )
            while batch := tuple(islice(bindings, 10)):
                await self._store.queue_deliveries(batch)
            return True
        path: Final = "/alerts?" + urlencode({"issue": pending.issue.id})
        issue_url: Final = activity_url(
            pending.alert.model_copy(update={"activity_path": path}), os.getenv("PROXY_BASE_URL")
        )
        source_url: Final = activity_url(pending.alert, os.getenv("PROXY_BASE_URL"))
        return await self._send(
            pending,
            urls,
            incident_message(pending.alert, None),
            f"LiteLLM alert: {pending.issue.title}",
            (("View alert", issue_url), ("View source activity", source_url)),
        )

    async def _send(
        self,
        pending: IssuePendingDelivery,
        urls: tuple[str, ...],
        message: str,
        subject: str,
        links: tuple[tuple[str, str], ...],
    ) -> bool:
        channel: Final = pending.delivery.channel
        recipients: Final = pending.delivery.email_recipients
        if channel not in configured_destinations(self._slack):
            return False
        if channel == "email":
            html: Final = f"<p>{escape(message).replace(chr(10), '<br>')}</p>" + "".join(
                f'<p><a href="{escape(url, quote=True)}">{escape(label)}</a></p>' for label, url in links
            )
            for recipient in recipients:
                await self._email(pending.delivery.id, recipient, subject, html)
            return bool(recipients)
        destination: Final = pending.delivery.slack_destination
        webhook: Final = next(
            (url for url in urls if destination is None or hashlib.sha256(url.encode()).hexdigest() == destination),
            None,
        )
        if webhook is None:
            return False
        await self._post(webhook, message + "\n" + "\n".join(f"{label}: {url}" for label, url in links))
        return True

    def _webhooks(self) -> tuple[str, ...]:
        overrides: Final = self._slack.alert_to_webhook_url or {}
        configured_url: Final = (
            overrides.get(AlertType.usage_alerts)
            or self._slack.default_webhook_url
            or os.getenv("SLACK_WEBHOOK_URL")
            or os.getenv("ALERTING_WEBHOOK_URL")
        )
        if not configured_url:
            return ()
        return (configured_url,) if isinstance(configured_url, str) else _URLS.validate_python(configured_url)

    @staticmethod
    def _email_finished(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()  # consume a late failure after the calling job was cancelled

    async def _email(self, key: str, recipient: str, subject: str, html: str) -> None:
        cls: Final = type(self)
        previous: Final = cls._email_task
        if previous is not None:
            if not previous.done():
                raise RuntimeError("An alert email is still in flight; retry later")
            if not previous.cancelled() and previous.exception() is None:
                if cls._email_store is not None and cls._email_key is not None:
                    await cls._email_store.mark_delivery_sent(cls._email_key)
                    if cls._email_task is not previous:
                        raise RuntimeError("Another alert email is still in flight; retry later")
                elif cls._email_key != key:
                    raise RuntimeError("A completed alert email is awaiting its receipt")
                if cls._email_key == key:
                    cls._clear_email(previous)
                    return previous.result()
        task: Final = asyncio.create_task(cls._send_email(recipient, subject, html))
        cls._email_task = task
        cls._email_key = key
        cls._email_store = self._store
        task.add_done_callback(cls._email_finished)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception:
            cls._clear_email(task)
            raise
        else:
            cls._clear_email(task)

    @classmethod
    def _clear_email(cls, task: asyncio.Task[None]) -> None:
        if cls._email_task is task:
            cls._email_task = None
            cls._email_key = None
            cls._email_store = None

    @staticmethod
    async def _send_email(recipient: str, subject: str, html: str) -> None:
        from litellm.proxy.utils import send_email

        await send_email(receiver_email=recipient, subject=subject, html=html, raise_on_error=True, attempt_timeout=30)

    async def _post(self, webhook: str, message: str) -> None:
        response: Final[Response] = await self._slack.async_http_handler.post(  # pyright: ignore[reportUnknownMemberType]  # legacy HTTP signature; supplied arguments are concrete
            url=webhook,
            headers={"Content-type": "application/json"},
            data=json.dumps({"text": escape(message, quote=False)}),
            timeout=10,
        )
        response.raise_for_status()
