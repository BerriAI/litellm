from datetime import date, datetime, timezone
from typing import Final

import httpx
import pytest

from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.integrations.SlackAlerting.usage_alerts import UsageAlertDelivery, activity_url, incident_message
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
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


def test_activity_link_retains_dashboard_mount_and_entity_query() -> None:
    assert activity_url(_ALERT, "https://gateway.example.com/") == (
        "https://gateway.example.com/ui/api-keys?key=reporting-key&detail_tab=activity"
    )
    assert activity_url(_ALERT, None) == _ALERT.activity_path


def test_email_summary_omits_the_plain_text_link_used_by_slack() -> None:
    summary: Final = incident_message(_ALERT, None)
    assert summary.endswith("1,000.00; baseline 200.00; threshold 600.00.")
    assert "View activity" not in summary
    assert incident_message(_ALERT, _ALERT.activity_path) == f"{summary}\nView activity: {_ALERT.activity_path}"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 503])
async def test_slack_ack_requires_successful_http_response(status: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", "https://gateway.example.com")
    requests: Final[list[httpx.Request]] = []

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status)

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(transport))
    slack: Final = SlackAlerting(
        alerting=["slack"], default_webhook_url="https://hooks.example.com/test", async_http_handler=handler
    )
    delivery: Final = UsageAlertDelivery(slack)
    try:
        if status == 200:
            assert await delivery.send(_ALERT, _RULE, "slack") is True
        else:
            with pytest.raises(httpx.HTTPStatusError):
                await delivery.send(_ALERT, _RULE, "slack")
        assert len(requests) == 1
        assert b"https://gateway.example.com/ui/api-keys?key=reporting-key&detail_tab=activity" in requests[0].content
    finally:
        await handler.close()
