import json

import pytest

import litellm
from litellm.llms.nebius.chat.transformation import NebiusConfig

TOKEN_FACTORY_API_BASE = "https://api.tokenfactory.nebius.com/v1"


def test_nebius_provider_uses_token_factory_api_base(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_BASE", raising=False)

    model, provider, _, api_base = litellm.get_llm_provider(
        model="nebius/moonshotai/Kimi-K3",
        api_key="test-key",
    )

    assert model == "moonshotai/Kimi-K3"
    assert provider == "nebius"
    assert api_base == TOKEN_FACTORY_API_BASE


def test_token_factory_api_base_resolves_nebius_provider():
    model, provider, _, api_base = litellm.get_llm_provider(
        model="moonshotai/Kimi-K3",
        api_base=TOKEN_FACTORY_API_BASE,
        api_key="test-key",
    )

    assert model == "moonshotai/Kimi-K3"
    assert provider == "nebius"
    assert api_base == TOKEN_FACTORY_API_BASE


def test_nebius_api_base_override(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_BASE", "https://custom.example/v1")

    _, _, _, api_base = litellm.get_llm_provider(
        model="nebius/moonshotai/Kimi-K3",
        api_key="test-key",
    )

    assert api_base == "https://custom.example/v1"


def test_nebius_maps_max_completion_tokens():
    optional_params = NebiusConfig().map_openai_params(
        non_default_params={"max_completion_tokens": 256},
        optional_params={},
        model="moonshotai/Kimi-K3",
        drop_params=False,
    )

    assert optional_params == {"max_tokens": 256}


def test_nebius_does_not_claim_unimplemented_provider_endpoints():
    from litellm.utils import ProviderConfigManager

    assert (
        ProviderConfigManager.get_provider_responses_api_config(provider="nebius")
        is None
    )


@pytest.mark.respx()
def test_nebius_completion_targets_token_factory(respx_mock, monkeypatch):
    monkeypatch.delenv("NEBIUS_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route = respx_mock.post(f"{TOKEN_FACTORY_API_BASE}/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "moonshotai/Kimi-K3",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello from Token Factory",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
            },
        },
        status_code=200,
    )

    response = litellm.completion(
        model="nebius/moonshotai/Kimi-K3",
        messages=[{"role": "user", "content": "Hello"}],
        api_key="test-key",
    )

    assert route.called
    assert response.choices[0].message.content == "Hello from Token Factory"
    assert route.calls[0].request.headers["authorization"] == "Bearer test-key"
    assert json.loads(route.calls[0].request.content)["model"] == "moonshotai/Kimi-K3"
