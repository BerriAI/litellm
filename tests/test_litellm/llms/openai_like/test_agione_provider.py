"""Tests for AGIone provider configuration and integration."""

import os

import pytest

import litellm


class TestAGIoneProviderConfig:
    """Test AGIone provider configuration."""

    def test_agione_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "AGIONE")
        assert LlmProviders.AGIONE.value == "agione"
        assert "agione" in litellm.provider_list

    def test_agione_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("agione")

        agione = JSONProviderRegistry.get("agione")
        assert agione is not None
        assert agione.base_url == "https://agione.pro/hyperone/xapi/api/v1"
        assert agione.api_key_env == "AGIONE_API_KEY"
        assert agione.api_base_env == "AGIONE_API_BASE"
        assert agione.supported_endpoints == [
            "/v1/chat/completions",
            "/v1/messages",
            "/v1/responses",
        ]
        assert JSONProviderRegistry.supports_responses_api("agione") is True

    def test_agione_responses_url(self):
        from litellm.llms.openai_like.dynamic_config import create_responses_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        agione = JSONProviderRegistry.get("agione")
        assert agione is not None

        config_cls = create_responses_config_class(agione)
        config = config_cls()

        assert config.get_complete_url(api_base=None, litellm_params={}) == "https://agione.pro/hyperone/xapi/api/v1/responses"

    def test_agione_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="agione/deepseek/deepseek-v3.2/0000n",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "deepseek/deepseek-v3.2/0000n"
        assert provider == "agione"
        assert api_base == "https://agione.pro/hyperone/xapi/api/v1"

    def test_agione_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="agione/deepseek/deepseek-v3.2/0000n",
            custom_llm_provider=None,
            api_base="https://custom.example.com/v1",
            api_key="test-key",
        )

        assert provider == "agione"
        assert api_base == "https://custom.example.com/v1"
        assert api_key == "test-key"

    def test_agione_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "agione-chat",
                    "litellm_params": {
                        "model": "agione/deepseek/deepseek-v3.2/0000n",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "agione-chat"


def test_agione_completion_live():
    """Optional live smoke test. Runs only when AGIONE_API_KEY is set."""
    api_key = os.environ.get("AGIONE_API_KEY")
    if not api_key or api_key == "PASTE_YOUR_NEW_KEY_HERE":
        pytest.skip("AGIONE_API_KEY not set")

    response = litellm.completion(
        model="agione/deepseek/deepseek-v3.2/0000n",
        messages=[
            {
                "role": "user",
                "content": "Reply exactly: litellm-agione-ok",
            }
        ],
        max_tokens=20,
    )

    assert response is not None
    assert response.choices
    assert response.choices[0].message.content is not None
