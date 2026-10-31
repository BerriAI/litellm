from typing import Final

import httpx
import pytest
import respx
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import Choices, ModelResponse

MODELS: Final = ("deepseek-v4.1-flash-1017", "gemma-4-31b", "deepseek-v4.1-flash-1017-abliterated")
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def test_hopper_uses_provider_credentials_and_allows_explicit_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOPPER_API_KEY", "hopper-environment-key")
    monkeypatch.setenv("HOPPER_API_BASE", "https://hopper.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    assert get_llm_provider(model="hopper/deepseek-v4.1-flash-1017") == (
        "deepseek-v4.1-flash-1017",
        "hopper",
        "hopper-environment-key",
        "https://hopper.test/v1",
    )
    assert get_llm_provider(
        model="hopper/deepseek-v4.1-flash-1017", api_key="explicit-key", api_base="https://override.test/v1"
    ) == ("deepseek-v4.1-flash-1017", "hopper", "explicit-key", "https://override.test/v1")


@pytest.mark.parametrize("model", MODELS)
def test_hopper_chat_routes_model_and_discounts_cached_input(model: str, local_model_cost_map: None) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.withhopper.com/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "READY"}, "finish_reason": "stop"}
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
            model=f"hopper/{model}",
            api_key="hopper-test-key",
            messages=[{"role": "user", "content": "Reply READY"}],
            reasoning_effort="low",
            max_tokens=64,
        )
    assert route.calls.last.request.headers["authorization"] == "Bearer hopper-test-key"
    body: Final = JSON_OBJECT.validate_json(route.calls.last.request.content)
    assert body["model"] == model
    assert body["reasoning_effort"] == "low"
    assert body["messages"] == [{"role": "user", "content": "Reply READY"}]
    assert isinstance(response, ModelResponse)
    assert isinstance(response.choices[0], Choices)
    assert response.choices[0].message.content == "READY"
    rates: Final = litellm.model_cost[f"hopper/{model}"]
    expected: Final = (
        20 * rates["input_cost_per_token"]
        + 80 * rates["cache_read_input_token_cost"]
        + 5 * rates["output_cost_per_token"]
    )
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_hopper_messages_preserves_thinking_and_cache_usage(model: str, local_model_cost_map: None) -> None:
    client: Final = AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport())
    try:
        with respx.mock() as upstream:
            route: Final = upstream.post("https://api.withhopper.com/v1/messages").respond(
                200,
                json={
                    "id": "msg-test",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [{"type": "text", "text": "READY"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 20, "cache_read_input_tokens": 80, "output_tokens": 5},
                },
            )
            response: Final = await litellm.anthropic.messages.acreate(
                model=f"hopper/{model}",
                api_key="hopper-test-key",
                max_tokens=64,
                messages=[{"role": "user", "content": "Reply READY"}],
                thinking={"type": "disabled"},
                client=client,
            )
        body: Final = JSON_OBJECT.validate_json(route.calls.last.request.content)
        assert route.calls.last.request.headers["authorization"] == "Bearer hopper-test-key"
        assert body["model"] == model
        assert body["thinking"] == {"type": "disabled"}
        assert response["content"] == [{"type": "text", "text": "READY"}]
        assert response["usage"] == {"input_tokens": 20, "cache_read_input_tokens": 80, "output_tokens": 5}
    finally:
        await client.close()
