"""Tests for Ambient provider configuration and integration."""

import litellm
from litellm import LlmProviders, Router
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.llms.openai_like.json_loader import JSONProviderRegistry


class TestAmbientProviderConfig:
    def test_ambient_in_provider_list(self):
        assert LlmProviders.AMBIENT.value == "ambient"
        assert "ambient" in litellm.provider_list

    def test_ambient_json_config_exists(self):
        ambient = JSONProviderRegistry.get("ambient")
        assert ambient is not None
        assert ambient.base_url == "https://api.ambient.xyz/v1"
        assert ambient.api_key_env == "AMBIENT_API_KEY"
        assert ambient.api_base_env == "AMBIENT_API_BASE"
        assert ambient.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_ambient_glm_provider_resolution(self):
        model, provider, _, api_base = get_llm_provider(
            model="ambient/zai-org/GLM-5.1-FP8",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )
        assert model == "zai-org/GLM-5.1-FP8"
        assert provider == "ambient"
        assert api_base == "https://api.ambient.xyz/v1"

    def test_ambient_kimi_provider_resolution(self):
        """Provider resolution survives multi-segment model names with slashes."""
        model, provider, _, api_base = get_llm_provider(
            model="ambient/moonshotai/kimi-k2.6",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )
        assert model == "moonshotai/kimi-k2.6"
        assert provider == "ambient"
        assert api_base == "https://api.ambient.xyz/v1"

    def test_ambient_router_config(self):
        router = Router(
            model_list=[
                {
                    "model_name": "glm-5.1",
                    "litellm_params": {
                        "model": "ambient/zai-org/GLM-5.1-FP8",
                        "api_key": "test-key",
                    },
                }
            ]
        )
        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "glm-5.1"

    def test_ambient_models_in_cost_map(self):
        """Load the local backup directly: litellm.model_cost is fetched from the
        upstream main-branch URL at import time and won't reflect in-repo edits."""
        cost_map = GetModelCostMap.load_local_model_cost_map()

        glm = cost_map["ambient/zai-org/GLM-5.1-FP8"]
        assert glm["litellm_provider"] == "ambient"
        assert glm["mode"] == "chat"
        assert glm["max_input_tokens"] == 202752
        assert glm["max_output_tokens"] == 131072
        assert glm["input_cost_per_token"] == 1.4e-06
        assert glm["output_cost_per_token"] == 4.4e-06
        assert glm["supports_tool_choice"] is True
        assert glm["supports_reasoning"] is True
        assert glm["supports_response_schema"] is True

        kimi = cost_map["ambient/moonshotai/kimi-k2.6"]
        assert kimi["litellm_provider"] == "ambient"
        assert kimi["mode"] == "chat"
        assert kimi["max_input_tokens"] == 262144
        assert kimi["max_output_tokens"] == 262144
        assert kimi["input_cost_per_token"] == 9.5e-07
        assert kimi["output_cost_per_token"] == 4e-06
        assert kimi["cache_read_input_token_cost"] == 2e-07
        assert kimi["supports_function_calling"] is True
        assert kimi["supports_tool_choice"] is True
        assert kimi["supports_reasoning"] is True
        assert kimi["supports_response_schema"] is True
        assert kimi["supports_prompt_caching"] is True
        assert kimi["supports_vision"] is True
