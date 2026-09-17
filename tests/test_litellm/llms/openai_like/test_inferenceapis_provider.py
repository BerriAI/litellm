"""
Tests for the Inference APIs provider configuration and integration.
"""

import json
from unittest.mock import patch

import httpx

import litellm


class TestInferenceAPIsProviderConfig:
    def test_inferenceapis_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "INFERENCEAPIS")
        assert LlmProviders.INFERENCEAPIS.value == "inferenceapis"
        assert "inferenceapis" in litellm.provider_list

    def test_inferenceapis_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("inferenceapis")

        provider = JSONProviderRegistry.get("inferenceapis")
        assert provider is not None
        assert provider.base_url == "https://api.inferenceapis.com/v1"
        assert provider.api_key_env == "INFERENCEAPIS_API_KEY"
        assert provider.param_mappings == {}

    def test_inferenceapis_supports_responses_api(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.supports_responses_api("inferenceapis")

    def test_inferenceapis_in_openai_compatible_providers(self):
        from litellm.constants import (
            openai_compatible_endpoints,
            openai_compatible_providers,
        )

        assert "inferenceapis" in openai_compatible_providers
        assert "https://api.inferenceapis.com/v1" in openai_compatible_endpoints

    def test_inferenceapis_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _api_key, api_base = get_llm_provider(
            model="inferenceapis/deepseek-ai/DeepSeek-V4-Flash",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "deepseek-ai/DeepSeek-V4-Flash"
        assert provider == "inferenceapis"
        assert api_base == "https://api.inferenceapis.com/v1"

    def test_inferenceapis_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        _model, provider, api_key, api_base = get_llm_provider(
            model="inferenceapis/openai/gpt-oss-120b",
            custom_llm_provider=None,
            api_base="https://proxy.example.com/v1",
            api_key="sk-test",
        )

        assert provider == "inferenceapis"
        assert api_base == "https://proxy.example.com/v1"
        assert api_key == "sk-test"

    def test_inferenceapis_url_autodetection(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        _model, provider, _api_key, api_base = get_llm_provider(
            model="zai-org/GLM-5.3",
            custom_llm_provider=None,
            api_base="https://api.inferenceapis.com/v1",
            api_key=None,
        )
        assert provider == "inferenceapis"
        assert api_base == "https://api.inferenceapis.com/v1"

    def test_inferenceapis_max_completion_tokens_passed_through(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("inferenceapis")
        assert provider is not None
        config = create_config_class(provider)()

        optional_params = config.map_openai_params(
            non_default_params={"max_completion_tokens": 256},
            optional_params={},
            model="openai/gpt-oss-120b",
            drop_params=False,
        )
        assert optional_params["max_completion_tokens"] == 256

    def test_inferenceapis_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "inferenceapis-chat",
                    "litellm_params": {
                        "model": "inferenceapis/deepseek-ai/DeepSeek-V4-Flash",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "inferenceapis-chat"


CHAT_RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "deepseek-ai/DeepSeek-V4-Flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "pong"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}


def _capture_requests(sent):
    def fake_send(self, request, **kwargs):
        sent.append(request)
        return httpx.Response(200, json=CHAT_RESPONSE, request=request)

    return fake_send


class TestInferenceAPIsRequests:
    def test_completion_sends_request_to_inferenceapis(self):
        sent = []
        with patch.object(httpx.Client, "send", _capture_requests(sent)):
            response = litellm.completion(
                model="inferenceapis/deepseek-ai/DeepSeek-V4-Flash",
                messages=[{"role": "user", "content": "ping"}],
                max_completion_tokens=7,
                api_key="sk-test",
            )

        assert response.choices[0].message.content == "pong"
        assert len(sent) == 1
        request = sent[0]
        assert request.method == "POST"
        assert str(request.url) == "https://api.inferenceapis.com/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer sk-test"
        body = json.loads(request.content)
        assert body["model"] == "deepseek-ai/DeepSeek-V4-Flash"
        assert body["max_completion_tokens"] == 7
        assert "max_tokens" not in body

    def test_completion_reads_key_and_base_from_env(self, monkeypatch):
        monkeypatch.setenv("INFERENCEAPIS_API_KEY", "sk-from-env")
        monkeypatch.setenv("INFERENCEAPIS_API_BASE", "https://proxy.example.com/v1")
        sent = []
        with patch.object(httpx.Client, "send", _capture_requests(sent)):
            litellm.completion(
                model="inferenceapis/openai/gpt-oss-120b",
                messages=[{"role": "user", "content": "ping"}],
            )

        assert len(sent) == 1
        request = sent[0]
        assert str(request.url) == "https://proxy.example.com/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer sk-from-env"
        assert json.loads(request.content)["model"] == "openai/gpt-oss-120b"
