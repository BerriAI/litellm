import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from tests.unit.llms.sail.helpers import (
    CACHED_TOKENS,
    COMPLETION_TOKENS,
    MODEL,
    PROMPT_TOKENS,
    SAIL_API_BASE,
    SpendCapture,
    cost_at,
    messages_body,
    sent_body,
)

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


@pytest.mark.asyncio
async def test_sail_streamed_messages_bill_the_reported_window(
    sail_env: None, respx_mock: respx.MockRouter, spend_capture: SpendCapture
) -> None:
    body: Final = messages_body()
    usage: Final = {
        "input_tokens": PROMPT_TOKENS - CACHED_TOKENS,
        "cache_read_input_tokens": CACHED_TOKENS,
        "output_tokens": COMPLETION_TOKENS,
        "service_tier": "balanced",
    }
    events: Final = (
        {"type": "message_start", "message": {**body, "content": [], "usage": {**usage, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": usage},
        {"type": "message_stop"},
    )
    stream: Final = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    respx_mock.post(f"{SAIL_API_BASE}/messages").mock(
        return_value=httpx.Response(200, content=stream, headers={"content-type": "text/event-stream"})
    )

    response: Final = await litellm.anthropic_messages(
        model=MODEL, messages=MESSAGES, max_tokens=16, stream=True, litellm_call_id=spend_capture.call_id
    )
    chunks: Final = tuple([chunk async for chunk in response])

    assert chunks
    assert await spend_capture.settled_cost() == pytest.approx(cost_at("_balanced"))
