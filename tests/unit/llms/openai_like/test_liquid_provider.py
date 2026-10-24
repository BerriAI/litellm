import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

LIQUID_API_BASE: Final = "https://router.inference.ai.exchange/v1"
LIQUID_MODEL_ID: Final = "liquid.auto"
LIQUID_MODEL: Final = f"liquid/{LIQUID_MODEL_ID}"


def _chat_completion_body() -> dict[str, object]:
    return {
        "id": "chatcmpl_liquid",
        "object": "chat.completion",
        "created": 1_789_550_000,
        "model": LIQUID_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Liquid"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
    }


def test_liquid_provider_is_registered_with_its_endpoints():
    provider: Final = JSONProviderRegistry.get("liquid")

    assert provider is not None
    assert provider.base_url == LIQUID_API_BASE
    assert set(provider.supported_endpoints) == {"/v1/chat/completions", "/v1/responses", "/v1/messages"}


def test_liquid_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIQUID_API_KEY", "liquid-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model=LIQUID_MODEL,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == LIQUID_MODEL_ID
    assert provider == "liquid"
    assert api_key == "liquid-env-key"
    assert api_base == LIQUID_API_BASE


def test_liquid_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIQUID_API_KEY", "liquid-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model=LIQUID_MODEL,
        custom_llm_provider=None,
        api_base="https://liquid.internal.example/v1",
        api_key="liquid-explicit-key",
    )

    assert provider == "liquid"
    assert api_key == "liquid-explicit-key"
    assert api_base == "https://liquid.internal.example/v1"


def test_liquid_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{LIQUID_API_BASE}/chat/completions").respond(200, json=_chat_completion_body())
        response: Final = litellm.completion(
            model=LIQUID_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="liquid-test-key",
        )

    request: Final[httpx.Request] = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer liquid-test-key"
    assert body["model"] == LIQUID_MODEL_ID
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Liquid"
