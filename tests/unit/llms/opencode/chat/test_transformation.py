import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.opencode.common_utils import go_headers


@pytest.fixture
def go_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    for route in ("chat/completions", "messages", "responses", "gemini"):
        monkeypatch.setitem(
            litellm.model_cost,
            f"opencode_go/fixture-{route.split('/')[0]}",
            {
                "litellm_provider": "opencode_go",
                "mode": "responses" if route == "responses" else "chat",
                "supported_endpoints": ["/v1beta/models" if route == "gemini" else f"/v1/{route}"],
                "max_output_tokens": 128000,
                "max_input_tokens": 1000000,
                "input_cost_per_token": 0.000002,
                "output_cost_per_token": 0.000005,
                "supports_function_calling": True,
            },
        )
    monkeypatch.delenv("OPENCODE_GO_API_BASE", raising=False)


def upstream_reply(route: str, model: str) -> Mapping[str, object]:
    if route == "gemini":
        return {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        }
    if route == "messages":
        return {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    if route == "responses":
        return {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "model": model,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_test",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hello"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("upstream", ["chat/completions", "messages", "responses", "gemini"])
@pytest.mark.parametrize("ingress", ["chat", "responses", "messages"])
async def test_all_routes_use_model_protocol_and_preserve_session(
    go_catalog: None, upstream: str, ingress: str
) -> None:
    model: Final = f"fixture-{upstream.split('/')[0]}"

    def respond(request: httpx.Request) -> httpx.Response:
        family: Final = "anthropic" if upstream == "messages" else "openai"
        expected_path: Final = (
            f"/inference/go/google/v1beta/models/{model}:generateContent"
            if upstream == "gemini"
            else f"/inference/go/{family}/v1/{upstream}"
        )
        assert request.url.path == expected_path
        assert request.headers["authorization"] == "Bearer fake-go-key"
        assert request.headers["x-opencode-session"] == "conversation-1"
        body: Final = json.loads(request.content)
        if upstream == "gemini":
            assert body["contents"][0]["parts"][0]["text"] == "hello"
        else:
            assert body["model"] == model
            assert "input" in body if upstream == "responses" else "messages" in body
        if upstream == "messages":
            assert body["max_tokens"] == 128000
        return httpx.Response(200, json=upstream_reply(upstream, model))

    with respx.mock(using="httpx") as transport:
        transport.post().mock(side_effect=respond)
        kwargs: Final = {
            "model": f"opencode_go/{model}",
            "api_key": "fake-go-key",
            "litellm_session_id": "conversation-1",
        }
        if ingress == "chat":
            response: Final = await litellm.acompletion(messages=[{"role": "user", "content": "hello"}], **kwargs)
            assert response.choices[0].message.content == "hello"
            assert response.usage.total_tokens == 15
        elif ingress == "responses":
            response_api: Final = await litellm.aresponses(input="hello", **kwargs)
            assert response_api.output[0].content[0].text == "hello"
            assert response_api.usage.total_tokens == 15
        else:
            message: Final = await litellm.anthropic.messages.acreate(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=128000,
                **kwargs,
            )
            assert message["content"][0]["text"] == "hello"
            assert message["usage"]["output_tokens"] == 5


def test_go_never_falls_back_to_other_provider_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    with pytest.raises(litellm.AuthenticationError, match="OPENCODE_API_KEY"):
        go_headers({}, {}, None, "fixture-model")


@pytest.mark.parametrize("input_tokens,expected", [(100, 0.00025), (101, 0.000504)])
def test_go_uses_registered_context_tiers_not_underlying_provider_prices(
    monkeypatch: pytest.MonkeyPatch,
    input_tokens: int,
    expected: float,
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "opencode_go/fixture-priced",
        {
            "litellm_provider": "opencode_go",
            "mode": "chat",
            "input_cost_per_token": 0.000002,
            "output_cost_per_token": 0.000005,
            "tiered_pricing": [
                {"range": [0, 100], "input_cost_per_token": 0.000002, "output_cost_per_token": 0.000005},
                {"range": [100, 1000], "input_cost_per_token": 0.000004, "output_cost_per_token": 0.000010},
            ],
        },
    )
    cost: Final = litellm.cost_per_token(
        model="opencode_go/fixture-priced",
        prompt_tokens=input_tokens,
        completion_tokens=10,
    )
    assert sum(cost) == pytest.approx(expected)


@pytest.mark.parametrize("suffix", ["", "/v1", "/v1/messages", "/v1/messages/"])
def test_messages_custom_base_does_not_duplicate_version_or_route(go_catalog: None, suffix: str) -> None:
    from litellm.llms.opencode.messages.transformation import OpenCodeGoMessagesConfig

    assert (
        OpenCodeGoMessagesConfig().get_complete_url(
            f"https://gateway.example/go{suffix}",
            None,
            "fixture-messages",
            {},
            {},
        )
        == "https://gateway.example/go/v1/messages"
    )


@pytest.mark.parametrize("base", [None, "https://gateway.example/go/v1"])
def test_discovery_returns_qualified_ids_and_accepts_response_metadata(base: str | None) -> None:
    url: Final = "https://gateway.example/go/v1/models" if base else "https://opencode.ai/inference/go/v1/models"
    with respx.mock(using="httpx") as transport:
        transport.get(url).respond(
            json={
                "object": "list",
                "data": [
                    {"id": "fixture-chat", "object": "model"},
                    {"id": "fixture-messages", "object": "model"},
                ],
            }
        )
        assert litellm.get_valid_models(
            check_provider_endpoint=True,
            custom_llm_provider="opencode_go",
            api_key="fake-go-key",
            api_base=base,
        ) == [
            "opencode_go/fixture-chat",
            "opencode_go/fixture-messages",
        ]
