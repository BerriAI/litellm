import json
from typing import Final

import pytest
import respx

import litellm
from litellm.types.utils import ModelResponse

MODELS: Final = ("deepseek-flash", "glm-5.3-flash")


@pytest.mark.parametrize("model", MODELS)
def test_siapi_sends_chat_and_prices_cached_usage(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIAPI_API_KEY", "siapi-test-key")
    model_id: Final = f"siapi/{model}"
    messages: Final = [{"role": "user", "content": "Reply with pong."}]

    with respx.mock() as upstream:
        chat: Final = upstream.post("https://siapi.ai/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_test",
                "created": 1780000000,
                "object": "chat.completion",
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 10,
                    "total_tokens": 130,
                    "prompt_tokens_details": {"cached_tokens": 100},
                },
            },
        )
        response: Final = litellm.completion(model=model_id, messages=messages)

    assert isinstance(response, ModelResponse)
    assert json.loads(chat.calls.last.request.content) == {"model": model, "messages": messages}
    assert chat.calls.last.request.headers["authorization"] == "Bearer siapi-test-key"
    pricing: Final = litellm.get_model_info(model_id)
    uncached_rate: Final = pricing["input_cost_per_token"]
    cached_rate: Final = pricing["cache_read_input_token_cost"]
    output_rate: Final = pricing["output_cost_per_token"]
    assert uncached_rate is not None and cached_rate is not None and output_rate is not None
    assert 0 < cached_rate < uncached_rate and output_rate > 0
    assert litellm.completion_cost(completion_response=response) == pytest.approx(
        20 * uncached_rate + 100 * cached_rate + 10 * output_rate,
    )


def test_siapi_model_picker_catalog() -> None:
    assert set(litellm.get_valid_models(custom_llm_provider="siapi", check_provider_endpoint=False)) == {
        f"siapi/{model}" for model in MODELS
    }


def test_siapi_never_uses_an_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIAPI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-provider-key")

    with respx.mock() as upstream, pytest.raises(litellm.AuthenticationError, match="SIAPI_API_KEY"):
        litellm.completion(model="siapi/deepseek-flash", messages=[{"role": "user", "content": "ping"}])

    assert not upstream.calls
