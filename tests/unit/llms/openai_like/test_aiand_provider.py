import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
from litellm.utils import ProviderConfigManager

AIAND_API_BASE = "https://api.aiand.com/v1"


def test_aiand_chat_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIAND_API_KEY", "test-key")

    provider_config = JSONProviderRegistry.get("aiand")
    assert provider_config is not None
    assert litellm.LlmProviders.AIAND.value == "aiand"
    assert "aiand" in litellm.openai_compatible_providers
    assert provider_config.base_url == AIAND_API_BASE
    assert provider_config.api_key_env == "AIAND_API_KEY"
    assert provider_config.api_base_env == "AIAND_API_BASE"

    model, provider, api_key, api_base = get_llm_provider("aiand/openai/gpt-oss-120b")
    assert model == "openai/gpt-oss-120b"
    assert provider == "aiand"
    assert api_key == "test-key"
    assert api_base == AIAND_API_BASE


def test_aiand_completion_request(respx_mock: respx.MockRouter):
    messages = [{"role": "user", "content": "Hello"}]
    route = respx_mock.post(f"{AIAND_API_BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "openai/gpt-oss-120b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )
    )

    litellm.completion(
        model="aiand/openai/gpt-oss-120b",
        api_key="test-key",
        messages=messages,
    )

    assert route.called
    request = route.calls.last.request
    assert str(request.url) == f"{AIAND_API_BASE}/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    request_body = json.loads(request.read())
    assert request_body["model"] == "openai/gpt-oss-120b"
    assert request_body["messages"] == messages


def test_aiand_responses_provider():
    config = ProviderConfigManager.get_provider_responses_api_config(
        provider="aiand",
        model="aiand/openai/gpt-oss-120b",
    )

    assert config is not None
    assert config.custom_llm_provider == "aiand"
    assert config.get_complete_url(api_base=None, litellm_params={}) == f"{AIAND_API_BASE}/responses"


def _aiand_catalog() -> frozenset[str]:
    return frozenset(name for name, info in litellm.model_cost.items() if info.get("litellm_provider") == "aiand")


def test_aiand_wildcard_lists_its_priced_models():
    catalog: Final = _aiand_catalog()

    assert catalog
    assert frozenset(get_known_models_from_wildcard("aiand/*")) == catalog


def test_aiand_calls_are_billed_from_their_cost_rows():
    catalog: Final = _aiand_catalog()
    assert catalog

    for model in sorted(catalog):
        row: Final = litellm.model_cost[model]
        prompt_cost, completion_cost = litellm.cost_per_token(model=model, prompt_tokens=1000, completion_tokens=500)

        assert prompt_cost == pytest.approx(1000 * row["input_cost_per_token"])
        assert completion_cost == pytest.approx(500 * row["output_cost_per_token"])
        assert prompt_cost > 0 and completion_cost > 0
