"""
Tests for the Meta Model API (Muse Spark) provider configuration and integration.
"""

import litellm


class TestMetaProviderConfig:
    def test_meta_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "META")
        assert LlmProviders.META.value == "meta"
        assert "meta" in litellm.provider_list

    def test_meta_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("meta")

        meta = JSONProviderRegistry.get("meta")
        assert meta is not None
        assert meta.base_url == "https://api.meta.ai/v1"
        assert meta.api_key_env == "META_API_KEY"
        assert meta.api_base_env == "META_API_BASE"

    def test_meta_supports_responses_api(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.supports_responses_api("meta")

    def test_meta_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "meta" in openai_compatible_providers

    def test_meta_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="meta/muse-spark-1.1",
            custom_llm_provider=None,
            api_base=None,
            api_key="sk-test",
        )

        assert model == "muse-spark-1.1"
        assert provider == "meta"
        assert api_base == "https://api.meta.ai/v1"

    def test_meta_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="meta/muse-spark-1.1",
            custom_llm_provider=None,
            api_base="https://custom.meta.ai/v1",
            api_key="sk-test",
        )

        assert provider == "meta"
        assert api_base == "https://custom.meta.ai/v1"
        assert api_key == "sk-test"

    def test_meta_url_autodetection(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="muse-spark-1.1",
            custom_llm_provider=None,
            api_base="https://api.meta.ai/v1",
            api_key=None,
        )
        assert provider == "meta"
        assert api_base == "https://api.meta.ai/v1"

    def test_meta_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "muse-spark",
                    "litellm_params": {
                        "model": "meta/muse-spark-1.1",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "muse-spark"


class TestMuseSparkModelInfo:
    def test_muse_spark_pricing_and_capabilities(self):
        info = litellm.get_model_info("meta/muse-spark-1.1")

        assert info["litellm_provider"] == "meta"
        assert info["input_cost_per_token"] == 1.25e-06
        assert info["output_cost_per_token"] == 4.25e-06
        assert info["cache_read_input_token_cost"] == 1.5e-07
        assert info["max_input_tokens"] == 1048576
        assert info["supports_reasoning"] is True
        assert info["supports_web_search"] is True
        assert info["supports_vision"] is True
        assert info["supports_function_calling"] is True
        assert info["supports_prompt_caching"] is True

    def test_muse_spark_cost_calculation(self):
        from litellm import completion_cost
        from litellm.types.utils import ModelResponse, Usage

        response = ModelResponse(
            model="muse-spark-1.1",
            usage=Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        )
        cost = completion_cost(
            completion_response=response,
            model="meta/muse-spark-1.1",
            custom_llm_provider="meta",
        )
        expected = 1000 * 1.25e-06 + 500 * 4.25e-06
        assert abs(cost - expected) < 1e-12
