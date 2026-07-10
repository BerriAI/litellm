"""
Tests for the Meta Model API (Muse Spark) provider configuration and integration.
"""

import pytest

import litellm


@pytest.fixture
def local_model_cost_map(monkeypatch):
    """Force the bundled cost map so muse-spark-1.1 resolves.

    muse-spark-1.1 is a day-0 model that ships only in the bundled backup; the
    default remote fetch of ``main`` does not carry it yet, so tests that read its
    model info must pin the local map instead of depending on network state or a
    leaked cost map from an earlier test.
    """
    original_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_cost
        litellm.get_model_info.cache_clear()


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


class TestMetaReasoningParams:
    def test_muse_spark_supports_reasoning_effort(self, local_model_cost_map):
        params = litellm.get_supported_openai_params(model="muse-spark-1.1", custom_llm_provider="meta")
        assert params is not None
        assert "reasoning_effort" in params

    def test_reasoning_effort_mapped_through(self, local_model_cost_map):
        cfg = litellm.ProviderConfigManager.get_provider_chat_config(
            model="muse-spark-1.1", provider=litellm.LlmProviders.META
        )
        assert cfg is not None
        mapped = cfg.map_openai_params(
            non_default_params={"reasoning_effort": "xhigh"},
            optional_params={},
            model="muse-spark-1.1",
            drop_params=False,
        )
        assert mapped["reasoning_effort"] == "xhigh"

    def test_reasoning_effort_gated_on_capability(self):
        """A meta model without reasoning metadata must not advertise reasoning_effort."""
        params = litellm.get_supported_openai_params(model="some-non-reasoning-model", custom_llm_provider="meta")
        assert params is not None
        assert "reasoning_effort" not in params


class TestMetaAnthropicMessages:
    def test_meta_resolves_native_messages_config(self):
        from litellm.llms.openai_like.messages.transformation import (
            JSONProviderAnthropicMessagesConfig,
        )

        cfg = litellm.ProviderConfigManager.get_provider_anthropic_messages_config(
            model="muse-spark-1.1", provider=litellm.LlmProviders.META
        )
        assert isinstance(cfg, JSONProviderAnthropicMessagesConfig)

    def test_json_provider_without_messages_endpoint_resolves_none(self):
        cfg = litellm.ProviderConfigManager.get_provider_anthropic_messages_config(
            model="some-model", provider=litellm.LlmProviders.PINSTRIPES
        )
        assert cfg is None

    def test_complete_url_defaults_to_meta_base(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        from litellm.llms.openai_like.messages.transformation import (
            JSONProviderAnthropicMessagesConfig,
        )

        provider = JSONProviderRegistry.get("meta")
        assert provider is not None
        cfg = JSONProviderAnthropicMessagesConfig(provider)

        url = cfg.get_complete_url(
            api_base=None,
            api_key="sk-test",
            model="muse-spark-1.1",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api.meta.ai/v1/messages"

        override_url = cfg.get_complete_url(
            api_base="https://custom.meta.ai/v1",
            api_key="sk-test",
            model="muse-spark-1.1",
            optional_params={},
            litellm_params={},
        )
        assert override_url == "https://custom.meta.ai/v1/messages"

    def test_api_key_resolved_from_env(self, monkeypatch):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        from litellm.llms.openai_like.messages.transformation import (
            JSONProviderAnthropicMessagesConfig,
        )

        monkeypatch.setenv("META_API_KEY", "sk-env-key")
        provider = JSONProviderRegistry.get("meta")
        assert provider is not None
        cfg = JSONProviderAnthropicMessagesConfig(provider)

        headers, _ = cfg.validate_anthropic_messages_environment(
            headers={},
            model="muse-spark-1.1",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )
        assert headers["authorization"] == "Bearer sk-env-key"
        assert headers["anthropic-version"] == "2023-06-01"


class TestMuseSparkModelInfo:
    def test_muse_spark_pricing_and_capabilities(self, local_model_cost_map):
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

    def test_muse_spark_cost_calculation(self, local_model_cost_map):
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
