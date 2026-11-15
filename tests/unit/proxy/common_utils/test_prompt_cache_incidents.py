from datetime import datetime, timedelta, timezone
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.models.alert import Alert
from litellm.proxy.common_utils.prompt_cache_incidents import (
    SlackPromptCacheNotifier,
    attach_prompt_cache_incidents,
    incident_text,
)
from litellm.router_utils.prompt_cache_observations.incidents import IncidentSlot, PromptCacheIncidents

_START: Final = datetime(2026, 1, 8, tzinfo=timezone.utc)
_ALERT: Final = Alert.model_validate(
    {
        "id": "7",
        "type": "prompt_cache",
        "resource_type": "deployment",
        "resource_id": "claude-east",
        "status": "open",
        "window_start": _START,
        "window_end": _START + timedelta(seconds=3),
        "created_at": _START,
        "updated_at": _START,
        "evidence": {
            "config_fingerprint": "config-1",
            "prefix_fingerprint": "prefix-1",
            "provider": "anthropic",
            "provider_model": "claude-test",
            "consecutive_misses": 3,
            "proven_write_tokens": 2048,
            "ttl_seconds": 300,
        },
    }
)


@pytest.mark.parametrize(
    ("settings", "prisma", "attached"),
    [
        ({"prompt_cache_alerts": True}, MagicMock(), True),
        ({"prompt_cache_alerts": True}, None, False),
        ({}, MagicMock(), False),
        ({"prompt_cache_alerts": "true"}, MagicMock(), False),
    ],
)
def test_incidents_are_saved_only_when_the_gateway_opts_in_and_has_a_database(
    settings: dict[str, object], prisma: MagicMock | None, attached: bool
) -> None:
    slot = IncidentSlot()
    attach_prompt_cache_incidents(settings, prisma, None, "https://gateway.example", slot=slot)
    assert isinstance(slot.current(), PromptCacheIncidents) is attached


def test_turning_the_setting_off_detaches_a_previous_recorder() -> None:
    slot = IncidentSlot()
    attach_prompt_cache_incidents({"prompt_cache_alerts": True}, MagicMock(), None, "", slot=slot)
    attach_prompt_cache_incidents({}, MagicMock(), None, "", slot=slot)
    assert slot.current() is None


def test_incident_text_names_the_deployment_model_and_links_the_alert_without_prompt_content() -> None:
    text = incident_text(_ALERT, "https://gateway.example")
    assert "claude-east" in text and "anthropic/claude-test" in text and "3 requests" in text
    assert "<https://gateway.example/ui/alerts?alert=7|Review the alert>" in text
    assert "prefix-1" not in text


class _Slack:
    def __init__(self, webhooks: tuple[str, ...], outcomes: dict[str, bool | Exception]) -> None:
        self.webhooks = webhooks
        self.outcomes = outcomes
        self.posted: tuple[tuple[str, str], ...] = ()

    def performance_alert_webhooks(self) -> tuple[str, ...]:
        return self.webhooks

    async def send_performance_alert_summary(self, webhook: str, message: str) -> bool:
        self.posted = (*self.posted, (webhook, message))
        outcome = self.outcomes[webhook]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


async def test_each_webhook_gets_one_post_and_one_failure_does_not_stop_the_others() -> None:
    slack = _Slack(("https://a", "https://b"), {"https://a": httpx.ConnectError("down"), "https://b": True})
    await SlackPromptCacheNotifier(slack, "https://gateway.example").notify(_ALERT)
    assert [webhook for webhook, _ in slack.posted] == ["https://a", "https://b"]
