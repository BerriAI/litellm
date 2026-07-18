"""
Tests for Tensormesh provider configuration and integration.
"""

import pytest

import litellm

TENSORMESH_MODELS = [
    "tensormesh/Qwen/Qwen3.5-397B-A17B-FP8",
    "tensormesh/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "tensormesh/Qwen/Qwen3.6-27B-FP8",
    "tensormesh/lukealonso/GLM-5.1-NVFP4-MTP",
    "tensormesh/deepseek-ai/DeepSeek-V4-Flash",
    "tensormesh/moonshotai/Kimi-K2.6",
    "tensormesh/MiniMaxAI/MiniMax-M2.5",
    "tensormesh/google/gemma-4-31B-it",
    "tensormesh/openai/gpt-oss-120b",
    "tensormesh/openai/gpt-oss-20b",
]


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

    def test_tensormesh_responses_api_enabled(self):
        """Tensormesh declares /v1/responses in supported_endpoints, so litellm
        resolves a responses config for it."""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        from litellm.utils import ProviderConfigManager

        assert JSONProviderRegistry.supports_responses_api("tensormesh") is True
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider="tensormesh",
            model="tensormesh/openai/gpt-oss-120b",
        )
        assert config is not None
        assert config.custom_llm_provider == "tensormesh"

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


class TestTensormeshCostMap:
    """The serverless models are registered in the cost map so LiteLLM can
    price requests and unblock tool-calling params on the JSON provider path."""

    @pytest.fixture(autouse=True)
    def _use_local_model_cost_map(self, monkeypatch):
        original_model_cost = litellm.model_cost
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        litellm.model_cost = litellm.get_model_cost_map(url="")
        litellm.get_model_info.cache_clear()
        try:
            yield
        finally:
            litellm.model_cost = original_model_cost
            litellm.get_model_info.cache_clear()

    def test_models_registered_with_capabilities(self):
        for model in TENSORMESH_MODELS:
            info = litellm.get_model_info(model)
            assert info["litellm_provider"] == "tensormesh"
            assert info["mode"] == "chat"
            assert litellm.supports_function_calling(model) is True, model
            assert litellm.supports_response_schema(model) is True, model
            assert litellm.model_cost[model]["supports_tool_choice"] is True, model
            assert litellm.model_cost[model]["supports_prompt_caching"] is True, model

    def test_reasoning_flag_matches_expected_set(self):
        reasoning_models = {
            "tensormesh/deepseek-ai/DeepSeek-V4-Flash",
            "tensormesh/Qwen/Qwen3.5-397B-A17B-FP8",
            "tensormesh/Qwen/Qwen3.6-27B-FP8",
            "tensormesh/lukealonso/GLM-5.1-NVFP4-MTP",
            "tensormesh/MiniMaxAI/MiniMax-M2.5",
            "tensormesh/moonshotai/Kimi-K2.6",
            "tensormesh/openai/gpt-oss-120b",
            "tensormesh/openai/gpt-oss-20b",
            "tensormesh/google/gemma-4-31B-it",
        }
        for model in TENSORMESH_MODELS:
            assert litellm.supports_reasoning(model) is (model in reasoning_models), model

    def test_cost_is_wired_and_cache_reads_are_free(self):
        prompt_cost, completion_cost = litellm.cost_per_token(
            model="tensormesh/openai/gpt-oss-120b",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        assert prompt_cost == pytest.approx(0.15)
        assert completion_cost == pytest.approx(0.60)
        assert (
            litellm.model_cost["tensormesh/openai/gpt-oss-120b"][
                "cache_read_input_token_cost"
            ]
            == 0
        )
