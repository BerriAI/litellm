import re
from typing import Final

import httpx
import pytest
import respx

import litellm
from tests.unit.llms.sail.helpers import (
    MODEL,
    SAIL_API_BASE,
    SpendCapture,
    chat_completion_stream,
    cost_at,
    sent_body,
)

MESSAGES: Final = [{"role": "user", "content": "hi"}]
TIER_CASES: Final = [
    pytest.param(None, None, "", id="no-tier"),
    pytest.param("auto", None, "", id="auto"),
    pytest.param("default", "asap", "", id="default"),
    pytest.param("priority", "asap", "", id="priority"),
    pytest.param("flex", "flex", "_flex", id="flex"),
    pytest.param("balanced", "balanced", "_balanced", id="balanced"),
    pytest.param("FLEX", "flex", "_flex", id="flex-any-case"),
]


def _window(body: dict[str, object]) -> object:
    metadata: Final = body.get("metadata")
    return metadata.get("completion_window") if isinstance(metadata, dict) else None


@pytest.mark.parametrize(("service_tier", "window", "column_suffix"), TIER_CASES)
@pytest.mark.asyncio
async def test_sail_chat_sends_the_tier_window_and_bills_its_price_columns(
    sail_env: None,
    chat_route: respx.Route,
    spend_capture: SpendCapture,
    service_tier: str | None,
    window: str | None,
    column_suffix: str,
) -> None:
    await litellm.acompletion(
        model=MODEL, messages=MESSAGES, service_tier=service_tier, litellm_call_id=spend_capture.call_id
    )

    body: Final = sent_body(chat_route)
    assert "service_tier" not in body
    assert _window(body) == window
    assert await spend_capture.settled_cost() == pytest.approx(cost_at(column_suffix))


@pytest.mark.parametrize(("service_tier", "window", "column_suffix"), TIER_CASES)
@pytest.mark.asyncio
async def test_sail_chat_stream_sends_the_tier_window_and_bills_its_price_columns(
    sail_env: None,
    respx_mock: respx.MockRouter,
    spend_capture: SpendCapture,
    service_tier: str | None,
    window: str | None,
    column_suffix: str,
) -> None:
    route: Final = respx_mock.post(f"{SAIL_API_BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200, content=chat_completion_stream(), headers={"content-type": "text/event-stream"}
        )
    )

    stream: Final = await litellm.acompletion(
        model=MODEL,
        messages=MESSAGES,
        service_tier=service_tier,
        stream=True,
        stream_options={"include_usage": True},
        litellm_call_id=spend_capture.call_id,
    )
    async for _ in stream:
        pass

    body: Final = sent_body(route)
    assert "service_tier" not in body
    assert _window(body) == window
    assert await spend_capture.settled_cost() == pytest.approx(cost_at(column_suffix))


@pytest.mark.parametrize(
    ("service_tier", "window"), [pytest.param(*case.values[:2], id=case.id) for case in TIER_CASES]
)
def test_sail_sync_chat_sends_the_tier_window(
    sail_env: None, chat_route: respx.Route, service_tier: str | None, window: str | None
) -> None:
    litellm.completion(model=MODEL, messages=MESSAGES, service_tier=service_tier)

    body: Final = sent_body(chat_route)
    assert "service_tier" not in body
    assert _window(body) == window


@pytest.mark.parametrize("service_tier", ["scale", "standard", "asap", 5, ["flex"]])
@pytest.mark.asyncio
async def test_sail_chat_rejects_a_tier_with_no_window_before_sending(
    sail_env: None, chat_route: respx.Route, service_tier: object
) -> None:
    with pytest.raises(litellm.UnsupportedParamsError, match=re.escape(f"service_tier={service_tier!r}")) as error:
        await litellm.acompletion(model=MODEL, messages=MESSAGES, service_tier=service_tier)

    assert error.value.status_code == 400
    assert not chat_route.called


@pytest.mark.parametrize("service_tier", ["scale", 5])
@pytest.mark.asyncio
async def test_sail_chat_drops_an_unknown_tier_under_drop_params_and_bills_asap(
    sail_env: None, chat_route: respx.Route, spend_capture: SpendCapture, service_tier: object
) -> None:
    await litellm.acompletion(
        model=MODEL,
        messages=MESSAGES,
        service_tier=service_tier,
        drop_params=True,
        litellm_call_id=spend_capture.call_id,
    )

    body: Final = sent_body(chat_route)
    assert "service_tier" not in body
    assert "metadata" not in body
    assert await spend_capture.settled_cost() == pytest.approx(cost_at(""))


@pytest.fixture(params=["openai-sdk", "base-http-handler"])
def chat_http_path(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", str(request.param == "base-http-handler"))


@pytest.mark.parametrize(
    ("service_tier", "wire_metadata", "column_suffix"),
    [
        pytest.param("flex", {"trace_id": "t-1", "completion_window": "flex"}, "_flex", id="flex"),
        pytest.param(None, {"trace_id": "t-1"}, "", id="no-tier"),
    ],
)
@pytest.mark.asyncio
async def test_sail_chat_merges_caller_extra_body_metadata_with_the_tier_window(
    sail_env: None,
    chat_http_path: None,
    chat_route: respx.Route,
    spend_capture: SpendCapture,
    service_tier: str | None,
    wire_metadata: dict[str, str],
    column_suffix: str,
) -> None:
    await litellm.acompletion(
        model=MODEL,
        messages=MESSAGES,
        service_tier=service_tier,
        extra_body={"metadata": {"trace_id": "t-1"}, "foo": 1},
        litellm_call_id=spend_capture.call_id,
    )

    body: Final = sent_body(chat_route)
    assert body["metadata"] == wire_metadata
    assert body["foo"] == 1
    assert await spend_capture.settled_cost() == pytest.approx(cost_at(column_suffix))


@pytest.mark.parametrize(
    ("extra_body", "message"),
    [
        pytest.param(
            {"metadata": {"completion_window": "flex"}},
            "extra_body.metadata.completion_window",
            id="extra-body-window",
        ),
        pytest.param({"service_tier": "flex"}, "service_tier inside extra_body", id="extra-body-tier"),
    ],
)
@pytest.mark.parametrize("service_tier", [None, "balanced"])
@pytest.mark.asyncio
async def test_sail_chat_rejects_a_window_billing_cannot_see_before_sending(
    sail_env: None,
    chat_http_path: None,
    chat_route: respx.Route,
    service_tier: str | None,
    extra_body: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(litellm.UnsupportedParamsError, match=message) as error:
        await litellm.acompletion(model=MODEL, messages=MESSAGES, service_tier=service_tier, extra_body=extra_body)

    assert error.value.status_code == 400
    assert not chat_route.called


@pytest.mark.parametrize(
    ("service_tier", "wire_metadata", "column_suffix"),
    [
        pytest.param("balanced", {"trace_id": "t-1", "completion_window": "balanced"}, "_balanced", id="balanced"),
        pytest.param(None, {"trace_id": "t-1"}, "", id="no-tier"),
    ],
)
@pytest.mark.asyncio
async def test_sail_chat_drops_a_window_billing_cannot_see_under_drop_params(
    sail_env: None,
    chat_http_path: None,
    chat_route: respx.Route,
    spend_capture: SpendCapture,
    service_tier: str | None,
    wire_metadata: dict[str, str],
    column_suffix: str,
) -> None:
    await litellm.acompletion(
        model=MODEL,
        messages=MESSAGES,
        service_tier=service_tier,
        extra_body={"service_tier": "flex", "metadata": {"trace_id": "t-1", "completion_window": "flex"}},
        drop_params=True,
        litellm_call_id=spend_capture.call_id,
    )

    body: Final = sent_body(chat_route)
    assert "service_tier" not in body
    assert body["metadata"] == wire_metadata
    assert await spend_capture.settled_cost() == pytest.approx(cost_at(column_suffix))


@pytest.mark.asyncio
async def test_sail_chat_drops_a_lone_caller_window_under_drop_params_and_bills_asap(
    sail_env: None, chat_http_path: None, chat_route: respx.Route, spend_capture: SpendCapture
) -> None:
    await litellm.acompletion(
        model=MODEL,
        messages=MESSAGES,
        extra_body={"metadata": {"completion_window": "flex"}},
        drop_params=True,
        litellm_call_id=spend_capture.call_id,
    )

    assert "completion_window" not in (sent_body(chat_route).get("metadata") or {})
    assert await spend_capture.settled_cost() == pytest.approx(cost_at(""))


@pytest.mark.asyncio
async def test_sail_chat_passes_a_non_mapping_extra_body_metadata_through_untouched(
    sail_env: None, chat_http_path: None, chat_route: respx.Route
) -> None:
    await litellm.acompletion(model=MODEL, messages=MESSAGES, extra_body={"metadata": None, "foo": 1})

    body: Final = sent_body(chat_route)
    assert "metadata" in body
    assert body["metadata"] is None
    assert body["foo"] == 1


def test_sail_sync_chat_rejects_an_unknown_tier_as_unsupported_params(sail_env: None, chat_route: respx.Route) -> None:
    with pytest.raises(litellm.UnsupportedParamsError, match="service_tier='scale'"):
        litellm.completion(model=MODEL, messages=MESSAGES, service_tier="scale")

    assert not chat_route.called


@pytest.mark.asyncio
async def test_sail_chat_keeps_the_window_when_preview_features_forward_caller_metadata(
    sail_env: None,
    chat_http_path: None,
    chat_route: respx.Route,
    spend_capture: SpendCapture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "enable_preview_features", True)

    await litellm.acompletion(
        model=MODEL,
        messages=MESSAGES,
        service_tier="flex",
        metadata={"requester_metadata": {"trace_id": "t-1"}},
        litellm_call_id=spend_capture.call_id,
    )

    assert sent_body(chat_route)["metadata"] == {"trace_id": "t-1", "completion_window": "flex"}
    assert await spend_capture.settled_cost() == pytest.approx(cost_at("_flex"))


@pytest.mark.parametrize(
    "rejected",
    [
        pytest.param({"stop": ["x"]}, id="stop"),
        pytest.param({"seed": 1}, id="seed"),
        pytest.param({"frequency_penalty": 0.5}, id="frequency_penalty"),
        pytest.param({"presence_penalty": 0.5}, id="presence_penalty"),
        pytest.param({"logit_bias": {"1": 1}}, id="logit_bias"),
        pytest.param({"logprobs": True}, id="logprobs"),
        pytest.param({"top_logprobs": 2}, id="top_logprobs"),
    ],
)
def test_sail_chat_rejects_params_sail_rejects_unless_dropped(
    sail_env: None, chat_route: respx.Route, rejected: dict[str, object]
) -> None:
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.completion(model=MODEL, messages=MESSAGES, **rejected)
    assert not chat_route.called

    litellm.completion(model=MODEL, messages=MESSAGES, drop_params=True, **rejected)
    assert set(rejected).isdisjoint(sent_body(chat_route))


def test_sail_chat_forwards_params_sail_accepts(sail_env: None, chat_route: respx.Route) -> None:
    tools: Final = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]

    litellm.completion(
        model=MODEL,
        messages=MESSAGES,
        max_tokens=64,
        tools=tools,
        tool_choice="auto",
        response_format={"type": "json_object"},
        reasoning_effort="low",
        user="user-1",
    )

    body: Final = sent_body(chat_route)
    assert body["max_tokens"] == 64
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"
    assert body["response_format"] == {"type": "json_object"}
    assert body["reasoning_effort"] == "low"
    assert body["user"] == "user-1"


def test_sail_chat_passes_max_tokens_and_max_completion_tokens_through_as_sent(
    sail_env: None, chat_route: respx.Route
) -> None:
    litellm.completion(model=MODEL, messages=MESSAGES, max_tokens=64, max_completion_tokens=32)

    body: Final = sent_body(chat_route)
    assert body["max_tokens"] == 64
    assert body["max_completion_tokens"] == 32


def test_sail_chat_uses_sail_api_base_env_and_key(
    sail_env: None, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SAIL_API_BASE", "https://sail-gateway.invalid/v1")
    route: Final = respx_mock.post("https://sail-gateway.invalid/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"id": "c", "object": "chat.completion", "created": 0, "model": "m", "choices": []}
        )
    )

    litellm.completion(model=MODEL, messages=MESSAGES)

    assert route.calls.last.request.headers["Authorization"] == "Bearer sail-test-key"
