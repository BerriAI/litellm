"""
Tests for PZERO provider configuration and integration.

PZERO (https://pzero.studio) is a prepaid inference marketplace (OpenAI chat
completions and Responses API) sitting in front of Venice's upstream, distinct
from the already-registered `veniceai` provider, which points at Venice's own
host directly. See GH #38632.
"""

import litellm


class TestPZEROProviderConfig:
    """Test PZERO provider configuration"""

    def test_pzero_in_provider_list(self):
        """Test that pzero is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "PZERO")
        assert LlmProviders.PZERO.value == "pzero"
        assert "pzero" in litellm.provider_list

    def test_pzero_json_config_exists(self):
        """Test that pzero is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("pzero")

        pzero = JSONProviderRegistry.get("pzero")
        assert pzero is not None
        assert pzero.base_url == "https://api.pzero.studio/v1"
        assert pzero.api_key_env == "PZERO_API_KEY"

    def test_pzero_in_openai_compatible_providers(self):
        """Test that pzero is in the openai_compatible_providers list"""
        from litellm.constants import openai_compatible_providers

        assert "pzero" in openai_compatible_providers

    def test_pzero_supports_responses_api(self):
        """PZERO exposes OpenAI Responses API, distinct from veniceai which doesn't."""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.supports_responses_api("pzero") is True

    def test_pzero_distinct_from_veniceai(self):
        """PZERO must not be conflated with the already-registered veniceai
        provider - they point at different hosts (see GH #38632)."""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        pzero = JSONProviderRegistry.get("pzero")
        veniceai = JSONProviderRegistry.get("veniceai")
        assert pzero is not None
        assert veniceai is not None
        assert pzero.base_url != veniceai.base_url
        assert pzero.api_key_env != veniceai.api_key_env

    def test_pzero_provider_resolution(self):
        """Test that provider resolution finds pzero and returns the default base URL"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="pzero/deepseek-v4-flash",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "deepseek-v4-flash"
        assert provider == "pzero"
        assert api_base == "https://api.pzero.studio/v1"

    def test_pzero_api_base_override(self):
        """Test that an explicit api_base / api_key overrides the default"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="pzero/deepseek-v4-flash",
            custom_llm_provider=None,
            api_base="https://custom.pzero.studio/v1",
            api_key="sk-test",
        )

        assert provider == "pzero"
        assert api_base == "https://custom.pzero.studio/v1"
        assert api_key == "sk-test"

    def test_pzero_url_autodetection(self):
        """Test that api_base=api.pzero.studio/v1 auto-sets custom_llm_provider=pzero"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="deepseek-v4-flash",
            custom_llm_provider=None,
            api_base="https://api.pzero.studio/v1",
            api_key=None,
        )
        assert provider == "pzero"
        assert api_base == "https://api.pzero.studio/v1"

    def test_pzero_router_config(self):
        """Test that pzero can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "pzero-chat",
                    "litellm_params": {
                        "model": "pzero/deepseek-v4-flash",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "pzero-chat"
