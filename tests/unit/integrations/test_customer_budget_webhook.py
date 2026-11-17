from datetime import datetime, timezone
from typing import Final

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.integrations.customer_budget_webhook import CustomerBudgetWebhook
from litellm.types.integrations.customer_budget import CustomerBudgetEvent

_URL: Final = "https://alerts.example/customer-budget"
_EVENT: Final = CustomerBudgetEvent(
    event="budget_crossed",
    scope="customer_member",
    entity_id="<!channel> <http://x|y> &",
    customer_org_id="org",
    customer_org_alias="tenant",
    customer_team_id="team",
    customer_member_id="<!channel> <http://x|y> &",
    spend=2,
    budget=1,
    period="lifetime",
    timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
)


@pytest.mark.asyncio
async def test_customer_webhook_posts_raw_event_and_configured_headers(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("CUSTOMER_BUDGET_WEBHOOK_URL", _URL)
    monkeypatch.setenv("CUSTOMER_BUDGET_WEBHOOK_HEADERS", '{"X-Env": "env", "X-Override": "env"}')
    callback: Final = CustomerBudgetWebhook.from_config({"headers": {"X-Override": "config"}})
    receiver: Final = respx_mock.post(_URL, json=_EVENT.model_dump(mode="json")).respond(200)
    await callback.async_on_customer_budget_event(_EVENT)
    assert receiver.call_count == 1
    assert receiver.calls[0].request.headers["X-Env"] == "env"
    assert receiver.calls[0].request.headers["X-Override"] == "config"


@pytest.mark.asyncio
async def test_customer_webhook_reports_non_success_response(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    receiver: Final = respx_mock.post(_URL).respond(503)
    callback: Final = CustomerBudgetWebhook(endpoint=_URL)
    with pytest.raises(httpx.HTTPStatusError):
        await callback.async_on_customer_budget_event(_EVENT)
    assert receiver.call_count == 1


@pytest.mark.parametrize("endpoint", (None, "", "not a url", "file:///tmp/events"))
def test_customer_webhook_requires_an_http_endpoint(monkeypatch: pytest.MonkeyPatch, endpoint: str | None) -> None:
    monkeypatch.delenv("CUSTOMER_BUDGET_WEBHOOK_URL", raising=False)
    with pytest.raises(ValidationError):
        CustomerBudgetWebhook(endpoint=endpoint)
