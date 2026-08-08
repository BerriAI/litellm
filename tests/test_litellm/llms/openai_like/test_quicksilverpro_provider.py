"""
Tests for QuickSilver Pro provider configuration and cost map registration.
"""

import pytest

import litellm

QUICKSILVERPRO_CHAT_MODELS = [
    "quicksilverpro/deepseek-v4-flash",
    "quicksilverpro/deepseek-v4-pro",
    "quicksilverpro/qwen3.8-max",
    "quicksilverpro/qwen3.7-max",
    "quicksilverpro/kimi-k3",
    "quicksilverpro/glm-5.2",
    "quicksilverpro/gpt-5.6-luna",
    "quicksilverpro/grok-4.5",
    "quicksilverpro/claude-haiku-4-5",
    "quicksilverpro/gemini-3.5-flash",
]


class TestQuickSilverProProviderConfig:
    """Test QuickSilver Pro provider configuration"""

    def test_quicksilverpro_json_config_exists(self):
        """Test that quicksilverpro is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("quicksilverpro")

        provider = JSONProviderRegistry.get("quicksilverpro")
        assert provider is not None
        assert provider.base_url == "https://api.quicksilverpro.io/v1"
        assert provider.api_key_env == "QUICKSILVERPRO_API_KEY"
        assert provider.api_base_env == "QUICKSILVERPRO_API_BASE"

    def test_quicksilverpro_provider_resolution(self):
        """Provider resolution finds quicksilverpro and its default base URL"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _api_key, api_base = get_llm_provider(
            model="quicksilverpro/qwen3.8-max",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "qwen3.8-max"
        assert provider == "quicksilverpro"
        assert api_base == "https://api.quicksilverpro.io/v1"

    def test_quicksilverpro_api_base_override(self):
        """An explicit api_base / api_key overrides the default"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        _model, provider, api_key, api_base = get_llm_provider(
            model="quicksilverpro/qwen3.8-max",
            custom_llm_provider=None,
            api_base="https://custom.example.com/v1",
            api_key="sk-test",
        )

        assert provider == "quicksilverpro"
        assert api_base == "https://custom.example.com/v1"
        assert api_key == "sk-test"

    def test_quicksilverpro_responses_api_enabled(self):
        """quicksilverpro declares /v1/responses, so litellm resolves a responses config"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        from litellm.utils import ProviderConfigManager

        assert JSONProviderRegistry.supports_responses_api("quicksilverpro") is True
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider="quicksilverpro",
            model="quicksilverpro/qwen3.8-max",
        )
        assert config is not None
        assert config.custom_llm_provider == "quicksilverpro"

    def test_quicksilverpro_router_config(self):
        """quicksilverpro can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "qsp-chat",
                    "litellm_params": {
                        "model": "quicksilverpro/qwen3.8-max",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "qsp-chat"


class TestQuickSilverProCostMap:
    """The published models are registered in the cost map so LiteLLM can price
    requests on the JSON provider path."""

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

    def test_models_registered_as_chat(self):
        for model in QUICKSILVERPRO_CHAT_MODELS:
            info = litellm.get_model_info(model)
            assert info["litellm_provider"] == "quicksilverpro", model
            assert info["mode"] == "chat", model
            assert litellm.supports_function_calling(model) is True, model
            assert litellm.supports_response_schema(model) is True, model

    def test_vision_models_declare_vision(self):
        """Only assert the positive case. litellm.supports_vision() falls back to a
        bare-model-name lookup, so a model id shared with another provider can report
        True from that entry rather than this one -- that is pre-existing helper
        behaviour, not something these entries assert."""
        vision_models = [
            "quicksilverpro/qwen3.8-max",
            "quicksilverpro/kimi-k3",
            "quicksilverpro/claude-haiku-4-5",
            "quicksilverpro/gemini-3.5-flash",
        ]
        for model in vision_models:
            assert litellm.get_model_info(model)["supports_vision"] is True, model

    def test_text_only_models_do_not_declare_vision(self):
        for model in ["quicksilverpro/deepseek-v4-flash", "quicksilverpro/glm-5.2"]:
            assert litellm.get_model_info(model).get("supports_vision") is not True, model

    def test_cost_per_token_matches_published_rates(self):
        """Rates are per 1M tokens on the published pricing page, stored per token here."""
        expected = {
            "quicksilverpro/deepseek-v4-flash": (0.112, 0.224),
            "quicksilverpro/qwen3.8-max": (2.0, 6.0),
            "quicksilverpro/kimi-k3": (2.4, 12.0),
            "quicksilverpro/claude-haiku-4-5": (0.8, 4.0),
        }
        for model, (in_1m, out_1m) in expected.items():
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=model, prompt_tokens=1_000_000, completion_tokens=1_000_000
            )
            assert prompt_cost == pytest.approx(in_1m), model
            assert completion_cost == pytest.approx(out_1m), model

    def test_prompt_caching_models_have_cache_read_cost(self):
        model = "quicksilverpro/deepseek-v4-flash"
        info = litellm.get_model_info(model)
        assert info["supports_prompt_caching"] is True
        assert litellm.model_cost[model]["cache_read_input_token_cost"] == pytest.approx(
            1.44e-08
        )

    def test_only_chat_models_are_registered(self):
        """Image generation dispatches from a hardcoded provider list that a
        declarative openai_like provider is not part of, so litellm.image_generation()
        would return an empty ImageResponse with no upstream request. No image entry
        is published until that path exists."""
        for info in self._entries().values():
            assert info["mode"] == "chat"

    def _entries(self):
        return {
            k: v
            for k, v in litellm.model_cost.items()
            if v.get("litellm_provider") == "quicksilverpro"
        }

    def test_every_entry_carries_a_source(self):
        entries = self._entries()
        assert len(entries) == 32
        for model, info in entries.items():
            assert info["source"] == "https://quicksilverpro.io/docs/models/", model
