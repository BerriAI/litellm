import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.sail.common_utils import billed_service_tier
from tests.unit.llms.sail.helpers import (
    MODEL,
    SAIL_API_BASE,
    SpendCapture,
    chat_completion_stream,
    cost_at,
    sent_body,
)

MESSAGES: Final = [{"role": "user", "content": "hi"}]


def _window(body: dict[str, object]) -> object:
    metadata: Final = body.get("metadata")
    return metadata.get("completion_window") if isinstance(metadata, dict) else None


@pytest.mark.parametrize(
    ("service_tier", "window", "column_suffix"),
    [
        pytest.param(None, None, "", id="none"),
        pytest.param("auto", None, "", id="auto"),
        pytest.param("default", "asap", "", id="default"),
        pytest.param("priority", "asap", "", id="priority"),
        pytest.param("flex", "flex", "_flex", id="flex"),
        pytest.param("balanced", "balanced", "_balanced", id="balanced"),
    ],
)
@pytest.mark.asyncio
async def test_sail_chat_sends_tier_window_and_bills_matching_price(
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


@pytest.mark.asyncio
async def test_sail_chat_stream_sends_metadata_window_and_bills_matching_price(
    sail_env: None, respx_mock: respx.MockRouter, spend_capture: SpendCapture
) -> None:
    route: Final = respx_mock.post(f"{SAIL_API_BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200, content=chat_completion_stream(), headers={"content-type": "text/event-stream"}
        )
    )

    stream: Final = await litellm.acompletion(
        model=MODEL,
        messages=MESSAGES,
        metadata={"completion_window": "flex"},
        stream=True,
        stream_options={"include_usage": True},
        litellm_call_id=spend_capture.call_id,
    )
    async for _ in stream:
        pass

    assert _window(sent_body(route)) == "flex"
    assert await spend_capture.settled_cost() == pytest.approx(cost_at("_flex"))


@pytest.mark.parametrize("preview", [False, True], ids=["preview-off", "preview-on"])
@pytest.mark.parametrize(
    ("caller_window", "window", "column_suffix"),
    [
        pytest.param("flex", "flex", "_flex", id="flex"),
        pytest.param("balanced", "balanced", "_balanced", id="balanced"),
        pytest.param("asap", "asap", "", id="asap"),
    ],
)
@pytest.mark.asyncio
async def test_sail_chat_accepts_caller_metadata_window_with_or_without_preview(
    sail_env: None,
    chat_route: respx.Route,
    spend_capture: SpendCapture,
    monkeypatch: pytest.MonkeyPatch,
    preview: bool,
    caller_window: str,
    window: str,
    column_suffix: str,
) -> None:
    monkeypatch.setattr(litellm, "enable_preview_features", preview)

    await litellm.acompletion(
        model=MODEL,
        messages=MESSAGES,
        metadata={"completion_window": caller_window},
        litellm_call_id=spend_capture.call_id,
    )

    body: Final = sent_body(chat_route)
    assert body["metadata"] == {"completion_window": window}
    assert "service_tier" not in body
    assert await spend_capture.settled_cost() == pytest.approx(cost_at(column_suffix))


@pytest.mark.parametrize("preview", [False, True], ids=["preview-off", "preview-on"])
@pytest.mark.asyncio
async def test_sail_chat_forwards_only_caller_metadata_from_proxy_shape(
    sail_env: None,
    chat_route: respx.Route,
    monkeypatch: pytest.MonkeyPatch,
    preview: bool,
) -> None:
    monkeypatch.setattr(litellm, "enable_preview_features", preview)
    caller: Final = {"completion_window": "flex", "trace_id": "t-1"}

    await litellm.acompletion(
        model=MODEL,
        messages=MESSAGES,
        metadata={**caller, "user_api_key_hash": "h", "requester_metadata": caller},
    )

    body: Final = sent_body(chat_route)
    assert "user_api_key_hash" not in json.dumps(body)
    assert body["metadata"] == caller


@pytest.mark.asyncio
async def test_sail_chat_does_not_forward_internal_proxy_metadata_without_caller_metadata(
    sail_env: None,
    chat_route: respx.Route,
) -> None:
    await litellm.acompletion(
        model=MODEL,
        messages=MESSAGES,
        metadata={
            "user_api_key_hash": "h",
            "user_api_key": "key",
            "requester_ip_address": "127.0.0.1",
            "api_base": SAIL_API_BASE,
            "requester_metadata": {},
        },
    )

    assert "metadata" not in sent_body(chat_route)


@pytest.mark.asyncio
async def test_openai_completion_does_not_forward_metadata(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setattr(litellm, "enable_preview_features", False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )

    await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"completion_window": "flex", "trace": "t"},
        api_key="sk-test",
    )

    assert route.called
    body: Final = json.loads(route.calls.last.request.content)
    assert "metadata" not in body


@pytest.mark.parametrize(
    ("service_tier", "caller_window"),
    [pytest.param("flex", "flex"), pytest.param("priority", "asap"), pytest.param("auto", "flex")],
)
@pytest.mark.asyncio
async def test_sail_chat_accepts_matching_tier_and_metadata_window(
    sail_env: None, chat_route: respx.Route, service_tier: str, caller_window: str
) -> None:
    await litellm.acompletion(
        model=MODEL,
        messages=MESSAGES,
        service_tier=service_tier,
        metadata={"completion_window": caller_window},
    )

    assert _window(sent_body(chat_route)) == caller_window


@pytest.mark.asyncio
async def test_sail_chat_rejects_conflicting_tier_and_metadata_window_before_dispatch(
    sail_env: None, chat_route: respx.Route
) -> None:
    with pytest.raises(litellm.UnsupportedParamsError, match="select different completion windows") as error:
        await litellm.acompletion(
            model=MODEL, messages=MESSAGES, service_tier="balanced", metadata={"completion_window": "flex"}
        )

    assert error.value.status_code == 400
    assert not chat_route.called


@pytest.mark.parametrize("drop_params", [None, "false"], ids=["unset", "string-false"])
@pytest.mark.asyncio
async def test_sail_chat_rejects_unknown_metadata_window_without_drop_params(
    sail_env: None, chat_route: respx.Route, drop_params: str | None
) -> None:
    with pytest.raises(litellm.UnsupportedParamsError, match=r"metadata\.completion_window") as error:
        await litellm.acompletion(
            model=MODEL, messages=MESSAGES, metadata={"completion_window": "scale"}, drop_params=drop_params
        )

    assert error.value.status_code == 400
    assert not chat_route.called


@pytest.mark.asyncio
async def test_sail_chat_drops_unknown_metadata_window_when_requested(
    sail_env: None, chat_route: respx.Route, spend_capture: SpendCapture
) -> None:
    await litellm.acompletion(
        model=MODEL,
        messages=MESSAGES,
        metadata={"completion_window": "scale"},
        drop_params=True,
        litellm_call_id=spend_capture.call_id,
    )

    assert _window(sent_body(chat_route)) is None
    assert await spend_capture.settled_cost() == pytest.approx(cost_at(""))


@pytest.mark.asyncio
async def test_sail_chat_merges_extra_body_metadata_with_normalized_window(
    sail_env: None, chat_route: respx.Route, spend_capture: SpendCapture
) -> None:
    await litellm.acompletion(
        model=MODEL,
        messages=MESSAGES,
        metadata={"completion_window": "flex"},
        extra_body={"metadata": {"trace": "t1"}, "foo": 1},
        litellm_call_id=spend_capture.call_id,
    )

    body: Final = sent_body(chat_route)
    assert body["metadata"] == {"trace": "t1", "completion_window": "flex"}
    assert body["foo"] == 1
    assert await spend_capture.settled_cost() == pytest.approx(cost_at("_flex"))


def test_sail_billed_service_tier_prefers_metadata_window() -> None:
    assert billed_service_tier({"service_tier": "balanced", "metadata": {"completion_window": "flex"}}) == "flex"


def test_sail_completion_cost_uses_metadata_window_for_billing() -> None:
    response: Final = {
        "id": "cost",
        "object": "chat.completion",
        "model": "zai-org/GLM-5.3",
        "choices": [],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
    }
    flex_cost: Final = litellm.completion_cost(
        completion_response=response,
        model=MODEL,
        custom_llm_provider="sail",
        optional_params={"metadata": {"completion_window": "flex"}},
    )
    tier_cost: Final = litellm.completion_cost(
        completion_response=response,
        model=MODEL,
        custom_llm_provider="sail",
        optional_params={"service_tier": "flex"},
    )
    default_cost: Final = litellm.completion_cost(
        completion_response=response,
        model=MODEL,
        custom_llm_provider="sail",
        optional_params={},
    )

    assert flex_cost == pytest.approx(tier_cost)
    assert flex_cost < default_cost
