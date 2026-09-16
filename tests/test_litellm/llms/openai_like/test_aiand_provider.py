import json

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
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
