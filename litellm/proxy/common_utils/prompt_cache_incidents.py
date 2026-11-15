from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, Protocol

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.models.alert import Alert, PromptCacheEvidence
from litellm.repositories.alert_repository import AlertRepository
from litellm.router_utils.prompt_cache_observations.incidents import (
    PROMPT_CACHE_INCIDENTS,
    IncidentSlot,
    PromptCacheIncidents,
)

if TYPE_CHECKING:
    from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
    from litellm.proxy.utils import PrismaClient

PROMPT_CACHE_ALERTS_SETTING: Final = "prompt_cache_alerts"
PROMPT_CACHE_ALERT_NOTIFICATIONS_SETTING: Final = "prompt_cache_alert_notifications"


def incident_text(alert: Alert, base_url: str) -> str:
    evidence: Final = alert.evidence
    model: Final = (
        f"{evidence.provider}/{evidence.provider_model}" if isinstance(evidence, PromptCacheEvidence) else "the model"
    )
    misses: Final = evidence.consecutive_misses if isinstance(evidence, PromptCacheEvidence) else 0
    return (
        f"*Prompt caching stopped on deployment {alert.resource_id}* ({model}): {misses} requests in a row wrote and "
        f"read no cache for a prefix the provider had cached. <{base_url}/ui/alerts?alert={alert.id}|Review the alert>"
    )


class SlackWebhooks(Protocol):
    def performance_alert_webhooks(self) -> tuple[str, ...]: ...

    async def send_performance_alert_summary(self, webhook: str, message: str) -> bool: ...


class SlackPromptCacheNotifier:
    """One post is attempted, for the request that opened the incident. A failed post is logged and not retried,
    so delivery is at most once and never guaranteed."""

    def __init__(self, slack: SlackWebhooks, base_url: str) -> None:
        self._slack: Final = slack
        self._base_url: Final = base_url

    async def _post(self, webhook: str, text: str) -> None:
        try:
            delivered: Final = await self._slack.send_performance_alert_summary(webhook, text)
        except httpx.HTTPError as error:
            verbose_proxy_logger.warning("prompt cache incident notification failed (%s)", type(error).__name__)
            return
        if not delivered:
            verbose_proxy_logger.warning("prompt cache incident notification was not accepted")

    async def notify(self, alert: Alert) -> None:
        text: Final = incident_text(alert, self._base_url)
        for webhook in self._slack.performance_alert_webhooks():
            await self._post(webhook, text)


def attach_prompt_cache_incidents(
    general_settings: Mapping[str, object],
    prisma_client: "PrismaClient | None",
    slack: "SlackAlerting | None",
    base_url: str,
    slot: IncidentSlot = PROMPT_CACHE_INCIDENTS,
) -> None:
    """Incidents are saved only when the gateway opts in and has a database. Otherwise deployments that opted
    into observation keep tracking streaks in Redis without saving anything. Slack is a separate opt-in reusing the
    performance alert webhooks, so subscribing to performance alerts never subscribes anyone to these."""
    if general_settings.get(PROMPT_CACHE_ALERTS_SETTING) is not True or prisma_client is None:
        slot.attach(None)
        return
    notified: Final = slack if general_settings.get(PROMPT_CACHE_ALERT_NOTIFICATIONS_SETTING) is True else None
    slot.attach(
        PromptCacheIncidents(
            AlertRepository(prisma_client),
            SlackPromptCacheNotifier(notified, base_url) if notified is not None else None,
        )
    )
