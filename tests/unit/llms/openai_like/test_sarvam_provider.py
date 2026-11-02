"""
Tests for the Sarvam provider configuration and cost map entries.
"""

import pytest

import litellm
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

SARVAM_MODELS = (
    "sarvam/sarvam-105b",
    "sarvam/sarvam-105b-conversations",
    "sarvam/glm5.3",
    "sarvam/glm5.3-flash",
    "sarvam/gemma4",
    "sarvam/deepseekv4.1-flash-1017",
)


class TestSarvamProviderConfig:
    def test_sarvam_json_config_exists(self):
        assert JSONProviderRegistry.exists("sarvam")

        sarvam = JSONProviderRegistry.get("sarvam")
        assert sarvam is not None
        assert sarvam.base_url == "https://api.sarvam.ai/v2"
        assert sarvam.api_key_env == "SARVAM_API_KEY"
        assert sarvam.base_class == "openai_gpt"
        assert "/v1/chat/completions" in sarvam.supported_endpoints
        assert "/v1/responses" in sarvam.supported_endpoints

    def test_sarvam_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="sarvam/sarvam-105b",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "sarvam-105b"
        assert provider == "sarvam"
        assert api_base == "https://api.sarvam.ai/v2"

    def test_sarvam_responses_api_enabled(self):
        from litellm.utils import ProviderConfigManager

        assert JSONProviderRegistry.supports_responses_api("sarvam") is True
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider="sarvam",
            model="sarvam/sarvam-105b",
        )
        assert config is not None
        assert config.custom_llm_provider == "sarvam"


class TestSarvamCostMap:
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

    @pytest.mark.parametrize("model", SARVAM_MODELS)
    def test_model_registered(self, model):
        assert model in litellm.model_cost

        entry = litellm.model_cost[model]
        assert entry["litellm_provider"] == "sarvam"
        assert entry["mode"] == "chat"

    @pytest.mark.parametrize("model", SARVAM_MODELS)
    def test_output_limit_matches_context_window(self, model):
        entry = litellm.model_cost[model]

        assert entry["max_output_tokens"] == entry["max_input_tokens"]
        assert entry["max_tokens"] == entry["max_output_tokens"]

    def test_stale_sarvam_m_absent(self):
        assert "sarvam/sarvam-m" not in litellm.model_cost
