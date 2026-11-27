import json
from typing import Final

import pytest
import respx

import litellm

_MODEL: Final = "zai-org/GLM-5.3-Flash"
_PROVIDER_MODEL: Final = f"openinference/{_MODEL}"
_MESSAGES: Final = [{"role": "user", "content": "Say hello"}]
_USAGE: Final = {
    "prompt_tokens": 20,
    "completion_tokens": 3,
    "total_tokens": 23,
    "prompt_tokens_details": {"cached_tokens": 12},
}


def test_openinference_routes_chat_and_prices_cached_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENINFERENCE_API_KEY", "openinference-test-key")
    monkeypatch.delenv("OPENINFERENCE_API_BASE", raising=False)
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.openinference.ai/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-openinference",
                "object": "chat.completion",
                "created": 1795819200,
                "model": _MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
                "usage": _USAGE,
            },
        )
        response: Final = litellm.completion(model=_PROVIDER_MODEL, messages=_MESSAGES)

    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer openinference-test-key"
    assert json.loads(request.content) == {"model": _MODEL, "messages": _MESSAGES}
    assert response.choices[0].message.content == "Hello"
    assert response.usage.prompt_tokens_details.cached_tokens == 12

    info: Final = litellm.get_model_info(_PROVIDER_MODEL)
    assert 0 < info["cache_read_input_token_cost"] < info["input_cost_per_token"]
    assert litellm.completion_cost(model=_PROVIDER_MODEL, completion_response=response) == pytest.approx(
        8 * info["input_cost_per_token"]
        + 12 * info["cache_read_input_token_cost"]
        + 3 * info["output_cost_per_token"]
    )


def test_openinference_catalog_models_are_available_to_router(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENINFERENCE_API_KEY", "openinference-test-key")
    models: Final = litellm.get_valid_models(custom_llm_provider="openinference")
    assert _PROVIDER_MODEL in models
    assert "openinference/deepseek-ai/DeepSeek-V4.1-Flash" in models
    assert "openinference/deepseek-ai/DeepSeek-V4.1-Flash-1017" in models
