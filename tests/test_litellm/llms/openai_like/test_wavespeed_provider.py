"""
Tests for the WaveSpeedAI LLM provider configuration and integration.
"""

import litellm


class TestWavespeedProviderConfig:
    def test_wavespeed_in_provider_list(self):
        from litellm import LlmProviders

        assert LlmProviders.WAVESPEED.value == "wavespeed"
        assert "wavespeed" in litellm.provider_list

    def test_wavespeed_json_config(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("wavespeed")
        assert provider is not None
        assert provider.base_url == "https://llm.wavespeed.ai/v1"
        assert provider.api_key_env == "WAVESPEED_API_KEY"
        assert provider.api_base_env == "WAVESPEED_API_BASE"
        assert JSONProviderRegistry.supports_responses_api("wavespeed")

    def test_wavespeed_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "wavespeed" in openai_compatible_providers

    def test_provider_prefixed_model_keeps_upstream_prefix(self):
        """WaveSpeed model ids are themselves `{provider}/{model}`, so only the routing prefix may be stripped."""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="wavespeed/anthropic/claude-opus-4.8",
            custom_llm_provider=None,
            api_base=None,
            api_key="sk-test",
        )

        assert model == "anthropic/claude-opus-4.8"
        assert provider == "wavespeed"
        assert api_key == "sk-test"
        assert api_base == "https://llm.wavespeed.ai/v1"

    def test_api_key_and_base_resolved_from_env(self, monkeypatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("WAVESPEED_API_KEY", "sk-env-key")
        monkeypatch.setenv("WAVESPEED_API_BASE", "https://proxy.internal/v1")

        _, provider, api_key, api_base = get_llm_provider(
            model="wavespeed/deepseek/deepseek-v4-flash",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert provider == "wavespeed"
        assert api_key == "sk-env-key"
        assert api_base == "https://proxy.internal/v1"

    def test_url_autodetection_from_api_base(self, monkeypatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("WAVESPEED_API_KEY", "sk-env-key")

        _, provider, api_key, api_base = get_llm_provider(
            model="glm-5",
            custom_llm_provider=None,
            api_base="https://llm.wavespeed.ai/v1",
            api_key=None,
        )

        assert provider == "wavespeed"
        assert api_key == "sk-env-key"

    def test_chat_completions_url(self):
        config = litellm.ProviderConfigManager.get_provider_chat_config(
            model="anthropic/claude-opus-4.8", provider=litellm.LlmProviders.WAVESPEED
        )
        assert config is not None
        assert (
            config.get_complete_url(
                api_base=None,
                api_key="sk-test",
                model="anthropic/claude-opus-4.8",
                optional_params={},
                litellm_params={},
            )
            == "https://llm.wavespeed.ai/v1/chat/completions"
        )

    def test_anthropic_messages_passthrough(self):
        from litellm.llms.openai_like.messages.transformation import (
            JSONProviderAnthropicMessagesConfig,
        )

        config = litellm.ProviderConfigManager.get_provider_anthropic_messages_config(
            model="anthropic/claude-opus-4.8", provider=litellm.LlmProviders.WAVESPEED
        )
        assert isinstance(config, JSONProviderAnthropicMessagesConfig)
        assert (
            config.get_complete_url(
                api_base=None,
                api_key="sk-test",
                model="anthropic/claude-opus-4.8",
                optional_params={},
                litellm_params={},
            )
            == "https://llm.wavespeed.ai/v1/messages"
        )


class TestWavespeedModelInfo:
    def test_claude_opus_pricing_and_capabilities(self):
        info = litellm.get_model_info("wavespeed/anthropic/claude-opus-4.8")

        assert info["litellm_provider"] == "wavespeed"
        assert info["input_cost_per_token"] == 5e-06
        assert info["output_cost_per_token"] == 2.5e-05
        assert info["cache_read_input_token_cost"] == 5e-07
        assert info["cache_creation_input_token_cost"] == 6.25e-06
        assert info["max_input_tokens"] == 872000
        assert info["max_output_tokens"] == 128000
        assert info["supports_function_calling"] is True
        assert info["supports_prompt_caching"] is True

    def test_gen_5_claude_entries_carry_upstream_thinking_flags(self):
        for model in ("wavespeed/anthropic/claude-fable-5", "wavespeed/anthropic/claude-sonnet-5"):
            entry = litellm.model_cost[model]

            assert entry["supports_adaptive_thinking"] is True
            assert entry["supports_sampling_params"] is False
            assert entry["supports_vision"] is True

    def test_non_tool_model_does_not_advertise_function_calling(self):
        info = litellm.get_model_info("wavespeed/aion-labs/aion-2.0")

        assert info["supports_function_calling"] is False
        assert info["supports_reasoning"] is True

    def test_cost_calculation(self):
        from litellm import completion_cost
        from litellm.types.utils import ModelResponse, Usage

        response = ModelResponse(
            model="anthropic/claude-opus-4.8",
            usage=Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        )
        cost = completion_cost(
            completion_response=response,
            model="wavespeed/anthropic/claude-opus-4.8",
            custom_llm_provider="wavespeed",
        )

        assert abs(cost - (1000 * 5e-06 + 500 * 2.5e-05)) < 1e-12

    def test_long_context_tier_pricing(self):
        from litellm import completion_cost
        from litellm.types.utils import ModelResponse, Usage

        response = ModelResponse(
            model="qwen/qwen3.6-flash",
            usage=Usage(prompt_tokens=300_000, completion_tokens=1_000, total_tokens=301_000),
        )
        cost = completion_cost(
            completion_response=response,
            model="wavespeed/qwen/qwen3.6-flash",
            custom_llm_provider="wavespeed",
        )

        assert abs(cost - (300_000 * 1e-06 + 1_000 * 4e-06)) < 1e-9
