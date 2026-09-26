from typing import Final

import httpx
import pytest
import respx

import litellm
from tests.unit.llms.sail.helpers import MODEL, SAIL_API_BASE, SpendCapture, cost_at, messages_body, sent_body

MESSAGES: Final = [{"role": "user", "content": "hi"}]


@pytest.fixture
def messages_route(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(f"{SAIL_API_BASE}/messages").mock(return_value=httpx.Response(200, json=messages_body()))


@pytest.mark.parametrize("service_tier", [None, "auto", "priority", "flex", "balanced", "scale"])
@pytest.mark.asyncio
async def test_sail_messages_send_no_window_and_bill_asap_whatever_the_tier(
    sail_env: None, messages_route: respx.Route, spend_capture: SpendCapture, service_tier: str | None
) -> None:
    await litellm.anthropic_messages(
        model=MODEL, messages=MESSAGES, max_tokens=16, service_tier=service_tier, litellm_call_id=spend_capture.call_id
    )

    body: Final = sent_body(messages_route)
    assert body["messages"] == MESSAGES
    assert "service_tier" not in body
    assert "completion_window" not in (body.get("metadata") or {})
    assert await spend_capture.settled_cost() == pytest.approx(cost_at(""))
