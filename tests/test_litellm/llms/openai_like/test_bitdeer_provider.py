"""
Tests for Bitdeer AI provider configuration and integration.
"""

import json
import os

import litellm

workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))


class TestBitdeerProviderConfig:
    """Test Bitdeer AI provider configuration"""

    def test_bitdeer_in_provider_list(self):
        """Test that bitdeer is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "BITDEER_AI")
        assert LlmProviders.BITDEER_AI.value == "bitdeer-ai"
        assert "bitdeer-ai" in litellm.provider_list

    def test_bitdeer_json_config_exists(self):
        """Test that bitdeer is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("bitdeer-ai")

        bitdeer_ai = JSONProviderRegistry.get("bitdeer-ai")
        assert bitdeer_ai is not None
        assert bitdeer_ai.base_url == "https://api-inference.bitdeer.ai/v1"
        assert bitdeer_ai.api_key_env == "BITDEER_API_KEY"
        assert bitdeer_ai.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_bitdeer_in_openai_compatible_providers(self):
        """Test that bitdeer is in the openai_compatible_providers list"""
        from litellm.constants import openai_compatible_providers

        assert "bitdeer-ai" in openai_compatible_providers

    def test_bitdeer_provider_resolution(self):
        """Test that provider resolution finds bitdeer and returns the default base URL"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="bitdeer-ai/moonshotai/Kimi-K3",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "moonshotai/Kimi-K3"
        assert provider == "bitdeer-ai"
        assert api_base == "https://api-inference.bitdeer.ai/v1"

    def test_bitdeer_api_base_override(self):
        """Test that an explicit api_base / api_key overrides the default"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="bitdeer-ai/moonshotai/Kimi-K3",
            custom_llm_provider=None,
            api_base="https://custom.bitdeer.ai/v1",
            api_key="sk-test",
        )

        assert provider == "bitdeer-ai"
        assert api_base == "https://custom.bitdeer.ai/v1"
        assert api_key == "sk-test"

    def test_bitdeer_url_autodetection(self):
        """Test that api_base=api-inference.bitdeer.ai/v1 auto-sets custom_llm_provider=bitdeer"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="moonshotai/Kimi-K3",
            custom_llm_provider=None,
            api_base="https://api-inference.bitdeer.ai/v1",
            api_key=None,
        )
        assert provider == "bitdeer-ai"
        assert api_base == "https://api-inference.bitdeer.ai/v1"

    def test_bitdeer_router_config(self):
        """Test that bitdeer can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "bitdeer-ai-kimi-k3",
                    "litellm_params": {
                        "model": "bitdeer-ai/moonshotai/Kimi-K3",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "bitdeer-ai-kimi-k3"

    def test_bitdeer_model_cost_map(self):
        """Test that bitdeer model pricing is registered correctly"""
        with open(
            os.path.join(workspace_path, "model_prices_and_context_window.json")
        ) as f:
            model_cost = json.load(f)

        expected_models = {
            "bitdeer-ai/moonshotai/Kimi-K3": (3e-06, 1.5e-05),
            "bitdeer-ai/zai-org/GLM-5.2": (1.4e-06, 4.4e-06),
            "bitdeer-ai/deepseek-ai/DeepSeek-V4-Flash": (4e-08, 8e-08),
        }
        for model, (input_cost, output_cost) in expected_models.items():
            assert model in model_cost
            assert model_cost[model]["litellm_provider"] == "bitdeer-ai"
            assert model_cost[model]["input_cost_per_token"] == input_cost
            assert model_cost[model]["output_cost_per_token"] == output_cost
