from datetime import date, datetime, timezone
from typing import Final

import httpx
import pytest

from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.integrations.SlackAlerting.usage_alerts import UsageAlertDelivery, activity_url, incident_message
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.proxy.issues import Alert, AlertCohort, AlertDelivery, AlertPendingDelivery
from litellm.types.proxy.usage_alerts import PerformanceSignal, AlertRule

_NOW: Final = datetime(2026, 10, 20, tzinfo=timezone.utc)
_RULE: Final = AlertRule(
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
_ALERT: Final = PerformanceSignal(
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


_ISSUE: Final = Alert(
    id="issue-example",
    cohort_key="cohort-example",
    kind="performance",
    family="traffic",
    title="Traffic increased",
    cohort=AlertCohort(scope_type="key", scope_id="reporting-key"),
    observed_on=_ALERT.day,
    observed_at=_NOW,
    range_start=_ALERT.baseline_start,
    range_end=_ALERT.day,
    grouping_reason="same_resource",
    status="open",
    first_seen=_NOW,
    last_seen=_NOW,
    updated_at=_NOW,
)
_PENDING: Final = AlertPendingDelivery(
    issue=_ISSUE,
    rule=_RULE,
    alert=_ALERT,
    delivery=AlertDelivery(id="delivery-example", issue_id=_ISSUE.id, channel="slack"),
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
            assert await delivery.send_issue(_PENDING) is True
        else:
            with pytest.raises(httpx.HTTPStatusError):
                await delivery.send_issue(_PENDING)
        assert len(requests) == 1
        assert (
            b"https://gateway.example.com/ui/api-keys?key=reporting-key&amp;detail_tab=activity" in requests[0].content
        )
    finally:
        await handler.close()


@pytest.mark.asyncio
async def test_issue_delivery_links_to_investigation_and_preserves_the_bound_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", "https://gateway.example.com")
    requests: Final[list[httpx.Request]] = []

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(transport))
    slack: Final = SlackAlerting(
        alerting=["slack"], default_webhook_url="https://hooks.example.com/test", async_http_handler=handler
    )
    try:
        assert await UsageAlertDelivery(slack).send_issue(_PENDING)
        assert len(requests) == 1
        assert b"https://gateway.example.com/ui/alerts?alert=issue-example" in requests[0].content
        assert (
            b"https://gateway.example.com/ui/api-keys?key=reporting-key&amp;detail_tab=activity" in requests[0].content
        )
        assert b"1,000.00; baseline 200.00" in requests[0].content
    finally:
        await handler.close()


@pytest.mark.asyncio
async def test_slack_source_labels_cannot_inject_mentions_or_links() -> None:
    requests: Final[list[httpx.Request]] = []

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(transport))
    slack: Final = SlackAlerting(
        alerting=["slack"], default_webhook_url="https://hooks.example.com/test", async_http_handler=handler
    )
    try:
        assert await UsageAlertDelivery(slack).send_issue(
            _PENDING.model_copy(
                update={
                    "alert": _ALERT.model_copy(
                        update={"rule_name": "<!channel> & <https://example.com|fake>", "scope_id": "<@someone>"}
                    )
                }
            )
        )
        assert b"<!channel>" not in requests[0].content
        assert b"&lt;!channel&gt; &amp; &lt;https://example.com|fake&gt;" in requests[0].content
        assert b"&lt;@someone&gt;" in requests[0].content
    finally:
        await handler.close()
