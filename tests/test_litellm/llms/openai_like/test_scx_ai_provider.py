"""
Tests for SCX.ai provider configuration and integration.
"""

import litellm


class TestSCXAIProviderConfig:
    def test_scx_ai_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "SCX_AI")
        assert LlmProviders.SCX_AI.value == "scx-ai"
        assert "scx-ai" in litellm.provider_list

    def test_scx_ai_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("scx-ai")

        scx = JSONProviderRegistry.get("scx-ai")
        assert scx is not None
        assert scx.base_url == "https://api.scx.ai/v1"
        assert scx.api_key_env == "SCX_API_KEY"
        assert scx.param_mappings.get("max_completion_tokens") == "max_tokens"
        assert scx.constraints.get("temperature_max") == 1.0

    def test_scx_ai_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "scx-ai" in openai_compatible_providers

    def test_scx_ai_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="scx-ai/gpt-oss-120b",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "gpt-oss-120b"
        assert provider == "scx-ai"
        assert api_base == "https://api.scx.ai/v1"

    def test_scx_ai_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="scx-ai/gpt-oss-120b",
            custom_llm_provider=None,
            api_base="https://custom.scx.ai/v1",
            api_key="sk-test",
        )

        assert provider == "scx-ai"
        assert api_base == "https://custom.scx.ai/v1"
        assert api_key == "sk-test"

    def test_scx_ai_url_autodetection(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="gpt-oss-120b",
            custom_llm_provider=None,
            api_base="https://api.scx.ai/v1",
            api_key=None,
        )
        assert provider == "scx-ai"
        assert api_base == "https://api.scx.ai/v1"

    def test_scx_ai_temperature_clamped_to_max(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("scx-ai")
        assert provider is not None
        config = create_config_class(provider)()

        optional_params = config.map_openai_params(
            non_default_params={"temperature": 1.7},
            optional_params={},
            model="gpt-oss-120b",
            drop_params=False,
        )
        assert optional_params["temperature"] == 1.0

        optional_params = config.map_openai_params(
            non_default_params={"temperature": 0.4},
            optional_params={},
            model="gpt-oss-120b",
            drop_params=False,
        )
        assert optional_params["temperature"] == 0.4

    def test_scx_ai_max_completion_tokens_mapped(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("scx-ai")
        assert provider is not None
        config = create_config_class(provider)()

        optional_params = config.map_openai_params(
            non_default_params={"max_completion_tokens": 256},
            optional_params={},
            model="gpt-oss-120b",
            drop_params=False,
        )
        assert optional_params["max_tokens"] == 256
        assert "max_completion_tokens" not in optional_params

    def test_scx_ai_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "scx-chat",
                    "litellm_params": {
                        "model": "scx-ai/gpt-oss-120b",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "scx-chat"
