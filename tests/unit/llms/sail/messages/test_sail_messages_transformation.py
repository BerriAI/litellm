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


@pytest.mark.asyncio
async def test_sail_messages_forward_the_callers_window_and_bill_its_price(
    sail_env: None, messages_route: respx.Route, spend_capture: SpendCapture
) -> None:
    await litellm.anthropic_messages(
        model=MODEL,
        messages=MESSAGES,
        max_tokens=16,
        metadata={"completion_window": "balanced", "trace_id": "caller-trace"},
        litellm_metadata={"user_api_key_hash": "proxy-private"},
        litellm_call_id=spend_capture.call_id,
    )

    assert sent_body(messages_route)["metadata"] == {"completion_window": "balanced", "trace_id": "caller-trace"}
    assert await spend_capture.settled_cost() == pytest.approx(cost_at("_balanced"))


@pytest.fixture
def messages_route(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(f"{SAIL_API_BASE}/messages").mock(return_value=httpx.Response(200, json=messages_body()))


@pytest.mark.parametrize(
    ("service_tier", "window", "suffix"),
    [(None, None, ""), ("auto", None, ""), ("priority", "asap", ""), ("balanced", "balanced", "_balanced")],
)
@pytest.mark.asyncio
async def test_sail_messages_send_and_bill_the_selected_window(
    sail_env: None,
    messages_route: respx.Route,
    spend_capture: SpendCapture,
    service_tier: str | None,
    window: str | None,
    suffix: str,
) -> None:
    await litellm.anthropic_messages(
        model=MODEL, messages=MESSAGES, max_tokens=16, service_tier=service_tier, litellm_call_id=spend_capture.call_id
    )

    body: Final = sent_body(messages_route)
    assert body["messages"] == MESSAGES
    assert "service_tier" not in body
    assert body.get("metadata") == ({"completion_window": window} if window else None)
    assert await spend_capture.settled_cost() == pytest.approx(cost_at(suffix))


@pytest.mark.parametrize("drop_params", [False, True])
@pytest.mark.parametrize("as_metadata", [False, True])
@pytest.mark.asyncio
async def test_sail_messages_reject_flex_before_sending(
    sail_env: None, messages_route: respx.Route, drop_params: bool, as_metadata: bool
) -> None:
    params: Final = {"metadata": {"completion_window": "flex"}} if as_metadata else {"service_tier": "flex"}
    with pytest.raises(litellm.UnsupportedParamsError, match="background=true"):
        await litellm.anthropic_messages(
            model=MODEL, messages=MESSAGES, max_tokens=16, drop_params=drop_params, **params
        )

    assert not messages_route.called


@pytest.mark.asyncio
async def test_sail_messages_reject_conflicting_window_and_tier(sail_env: None, messages_route: respx.Route) -> None:
    with pytest.raises(litellm.UnsupportedParamsError, match="different completion windows"):
        await litellm.anthropic_messages(
            model=MODEL,
            messages=MESSAGES,
            max_tokens=16,
            service_tier="priority",
            metadata={"completion_window": "balanced"},
        )

    assert not messages_route.called


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
