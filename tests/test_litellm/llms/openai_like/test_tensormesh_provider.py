"""
Tests for Tensormesh provider configuration and integration.
"""

import litellm


class TestTensormeshProviderConfig:
    """Test Tensormesh provider configuration"""

    def test_tensormesh_in_provider_list(self):
        """Test that tensormesh is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "TENSORMESH")
        assert LlmProviders.TENSORMESH.value == "tensormesh"
        assert "tensormesh" in litellm.provider_list

    def test_tensormesh_json_config_exists(self):
        """Test that tensormesh is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("tensormesh")

        tensormesh = JSONProviderRegistry.get("tensormesh")
        assert tensormesh is not None
        assert tensormesh.base_url == "https://serverless.tensormesh.ai/v1"
        assert tensormesh.api_key_env == "TENSORMESH_INFERENCE_API_KEY"
        assert tensormesh.api_base_env == "TENSORMESH_SERVERLESS_BASE_URL"
        assert tensormesh.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_tensormesh_provider_resolution(self):
        """Test that provider resolution finds tensormesh and the default base URL"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="tensormesh/openai/gpt-oss-120b",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "openai/gpt-oss-120b"
        assert provider == "tensormesh"
        assert api_base == "https://serverless.tensormesh.ai/v1"

    def test_tensormesh_api_base_override(self):
        """Test that an explicit api_base / api_key overrides the serverless default"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="tensormesh/openai/gpt-oss-120b",
            custom_llm_provider=None,
            api_base="https://custom.example.com/v1",
            api_key="sk-test",
        )

        assert provider == "tensormesh"
        assert api_base == "https://custom.example.com/v1"
        assert api_key == "sk-test"

    def test_tensormesh_text_completion_enabled(self):
        """Tensormesh is wired for the /completions (text completion) route,
        matching the text_completion flag in provider_endpoints_support.json."""
        assert "tensormesh" in litellm.openai_text_completion_compatible_providers

    def test_tensormesh_router_config(self):
        """Test that tensormesh can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "tensormesh-chat",
                    "litellm_params": {
                        "model": "tensormesh/openai/gpt-oss-120b",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "tensormesh-chat"
