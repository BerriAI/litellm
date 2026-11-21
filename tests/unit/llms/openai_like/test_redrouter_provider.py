import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


def test_redrouter_resolves_environment_and_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDROUTER_API_KEY", "fake-environment-key")
    assert get_llm_provider(model="redrouter/red-cyber-1") == (
        "red-cyber-1",
        "redrouter",
        "fake-environment-key",
        "https://redrouter.org/v1",
    )
    assert get_llm_provider(
        model="redrouter/red-cyber-1", api_key="fake-explicit-key", api_base="https://gateway.example/v1"
    ) == ("red-cyber-1", "redrouter", "fake-explicit-key", "https://gateway.example/v1")


def test_redrouter_chat_forwards_reasoning_and_accounts_for_cache(local_model_cost_map: None) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://redrouter.org/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "red-cyber-1",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "total_tokens": 105,
                    "prompt_tokens_details": {"cached_tokens": 80},
                },
            },
        )
        response: Final = litellm.completion(
            model="redrouter/red-cyber-1",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "Say hello"}],
            reasoning_effort="low",
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert body["model"] == "red-cyber-1"
    assert body["reasoning_effort"] == "low"
    assert response.choices[0].message.content == "hello"
    rates: Final = litellm.model_cost["redrouter/red-cyber-1"]
    expected: Final = (
        20 * rates["input_cost_per_token"]
        + 80 * rates["cache_read_input_token_cost"]
        + 5 * rates["output_cost_per_token"]
    )
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


def test_redrouter_uses_native_responses_endpoint() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://redrouter.org/v1/responses").respond(
            200,
            json={
                "id": "resp-test",
                "object": "response",
                "created_at": 1,
                "model": "red-cyber-1",
                "status": "completed",
                "output": [
                    {
                        "id": "msg-test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="redrouter/red-cyber-1",
            api_key="fake-test-key",
            input="Say hello",
            store=False,
        )
    assert route.calls.last.request.headers["authorization"] == "Bearer fake-test-key"
    assert json.loads(route.calls.last.request.content)["model"] == "red-cyber-1"
    assert response.output[0].content[0].text == "hello"


@pytest.mark.asyncio
async def test_redrouter_messages_use_native_endpoint_and_preserve_output_cap() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://redrouter.org/v1/messages").respond(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "red-cyber-1",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="redrouter/red-cyber-1",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="fake-test-key",
            client=AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport()),
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert body["model"] == "red-cyber-1"
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "hello"
