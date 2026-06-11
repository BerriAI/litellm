"""
Tests for Hanzo provider configuration and integration.
"""

import pytest

import litellm

HANZO_MODELS = [
    "hanzo/zen4",
    "hanzo/zen4-max",
]


class TestHanzoProviderConfig:
    """Test Hanzo provider configuration"""

    def test_hanzo_in_provider_list(self):
        """Test that hanzo is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "HANZO")
        assert LlmProviders.HANZO.value == "hanzo"
        assert "hanzo" in litellm.provider_list

    def test_hanzo_json_config_exists(self):
        """Test that hanzo is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("hanzo")

        hanzo = JSONProviderRegistry.get("hanzo")
        assert hanzo is not None
        assert hanzo.base_url == "https://api.hanzo.ai/v1"
        assert hanzo.api_key_env == "HANZO_API_KEY"
        assert hanzo.api_base_env == "HANZO_API_BASE"
        assert hanzo.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_hanzo_provider_resolution(self):
        """Test that provider resolution finds hanzo and the default base URL"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="hanzo/zen4",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "zen4"
        assert provider == "hanzo"
        assert api_base == "https://api.hanzo.ai/v1"

    def test_hanzo_api_base_override(self):
        """Test that an explicit api_base / api_key overrides the default"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="hanzo/zen4",
            custom_llm_provider=None,
            api_base="https://custom.example.com/v1",
            api_key="sk-test",
        )

        assert provider == "hanzo"
        assert api_base == "https://custom.example.com/v1"
        assert api_key == "sk-test"

    def test_hanzo_router_config(self):
        """Test that hanzo can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "hanzo-chat",
                    "litellm_params": {
                        "model": "hanzo/zen4",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "hanzo-chat"


class TestHanzoCostMap:
    """Hanzo Zen models are registered in the cost map so LiteLLM can
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
        for model in HANZO_MODELS:
            info = litellm.get_model_info(model)
            assert info["litellm_provider"] == "hanzo"
            assert info["mode"] == "chat"
            assert litellm.supports_function_calling(model) is True, model
            assert litellm.supports_response_schema(model) is True, model
            assert litellm.model_cost[model]["supports_tool_choice"] is True, model

    def test_zen4_cost_is_wired(self):
        prompt_cost, completion_cost = litellm.cost_per_token(
            model="hanzo/zen4",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        assert prompt_cost == pytest.approx(1.5)
        assert completion_cost == pytest.approx(4.5)

    def test_zen4_max_cost_is_wired(self):
        prompt_cost, completion_cost = litellm.cost_per_token(
            model="hanzo/zen4-max",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        assert prompt_cost == pytest.approx(3.0)
        assert completion_cost == pytest.approx(12.0)

    def test_zen4_max_long_context_window(self):
        info = litellm.get_model_info("hanzo/zen4-max")
        assert info["max_input_tokens"] == 1_000_000
