"""
Tests for the PowerTokens JSON-configured provider.
"""

import litellm


class TestPowerTokensProviderConfig:
    def test_powertokens_in_provider_list(self):
        from litellm import LlmProviders

        assert LlmProviders.POWERTOKENS.value == "powertokens"
        assert "powertokens" in litellm.provider_list

    def test_powertokens_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("powertokens")
        powertokens = JSONProviderRegistry.get("powertokens")
        assert powertokens is not None
        assert powertokens.base_url == "https://api.powertokens.ai/v1"
        assert powertokens.api_key_env == "POWERTOKENS_API_KEY"
        assert powertokens.api_base_env == "POWERTOKENS_API_BASE"

    def test_powertokens_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="powertokens/glm-5.2",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "glm-5.2"
        assert provider == "powertokens"
        assert api_base == "https://api.powertokens.ai/v1"

    def test_powertokens_complete_url(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        config = create_config_class(JSONProviderRegistry.get("powertokens"))()
        url = config.get_complete_url(
            api_base="https://api.powertokens.ai/v1",
            api_key="test-key",
            model="powertokens/glm-5.2",
            optional_params={},
            litellm_params={},
            stream=False,
        )

        assert url == "https://api.powertokens.ai/v1/chat/completions"

    def test_powertokens_provider_config_manager(self):
        from litellm import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(
            model="glm-5.2", provider=LlmProviders.POWERTOKENS
        )

        assert config is not None
        assert config.custom_llm_provider == "powertokens"
