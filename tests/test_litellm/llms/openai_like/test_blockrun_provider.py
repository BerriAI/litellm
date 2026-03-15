"""
Tests for BlockRun provider configuration.
"""

import os
import sys

workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)

import litellm


class TestBlockRunProvider:
    """Test BlockRun JSON provider loading, resolution, and parameter mapping"""

    def test_blockrun_provider_loaded(self):
        """Test that BlockRun provider is loaded from providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("blockrun")
        config = JSONProviderRegistry.get("blockrun")
        assert config is not None
        assert config.base_url == "https://blockrun.ai/api/v1"
        assert config.api_key_env == "BLOCKRUN_WALLET_KEY"
        assert config.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_blockrun_provider_resolution(self):
        """Test that blockrun/ prefix resolves correctly"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="blockrun/openai/gpt-4o",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "openai/gpt-4o"
        assert provider == "blockrun"
        assert api_base == "https://blockrun.ai/api/v1"

    def test_blockrun_in_llm_providers_enum(self):
        """Test that BlockRun is in the LlmProviders enum"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "BLOCKRUN")
        assert LlmProviders.BLOCKRUN.value == "blockrun"
        assert "blockrun" in litellm.provider_list

    def test_blockrun_in_providers_set(self):
        """Test that blockrun is in LlmProvidersSet for param mapping"""
        from litellm.types.utils import LlmProvidersSet

        assert "blockrun" in LlmProvidersSet

    def test_blockrun_config_manager(self):
        """Test that ProviderConfigManager returns config for BlockRun"""
        from litellm import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(
            model="openai/gpt-4o", provider=LlmProviders.BLOCKRUN
        )

        assert config is not None
        assert config.custom_llm_provider == "blockrun"

    def test_blockrun_parameter_mapping(self):
        """Test that max_completion_tokens is mapped to max_tokens"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("blockrun")
        config_class = create_config_class(provider)
        config = config_class()

        optional_params = {}
        non_default_params = {"max_completion_tokens": 100, "temperature": 0.7}
        result = config.map_openai_params(
            non_default_params, optional_params, "openai/gpt-4o", False
        )

        assert "max_tokens" in result
        assert result["max_tokens"] == 100
        assert "max_completion_tokens" not in result
        assert result["temperature"] == 0.7

    def test_blockrun_dynamic_config(self):
        """Test dynamic config class creation for BlockRun"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("blockrun")
        config_class = create_config_class(provider)
        config = config_class()

        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://blockrun.ai/api/v1"

    def test_blockrun_free_model_resolution(self):
        """Test that free models also resolve correctly"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="blockrun/nvidia/gpt-oss-120b",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "nvidia/gpt-oss-120b"
        assert provider == "blockrun"

    def test_blockrun_router_config(self):
        """Test that blockrun can be used in Router configuration (fixes Router/Proxy support)"""
        from litellm import Router

        # This should not raise "Unsupported provider - blockrun"
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {
                        "model": "blockrun/openai/gpt-4o",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "gpt-4o"
