"""Tests for decoupling Azure deployment IDs from underlying model names.

When users name their Azure deployment something non-standard (e.g. "my-deployment-id"),
setting ``base_model`` should drive model-type detection (o-series, gpt-5,
etc.) so the correct config, supported params, and param mapping are used.
"""

import pytest

import litellm
from litellm.llms.azure.chat.gpt_5_transformation import AzureOpenAIGPT5Config
from litellm.llms.azure.chat.o_series_transformation import AzureOpenAIO1Config
from litellm.utils import ProviderConfigManager, get_optional_params


# ---------------------------------------------------------------------------
# _get_azure_config — routes to the correct config based on base_model
# ---------------------------------------------------------------------------
class TestGetAzureConfigWithBaseModel:
    """ProviderConfigManager._get_azure_config should use base_model for detection."""

    def test_should_return_gpt5_config_when_base_model_is_gpt5(self):
        config = ProviderConfigManager._get_azure_config(
            model="my-deployment-id", base_model="azure/gpt-5.2"
        )
        assert isinstance(config, AzureOpenAIGPT5Config)

    def test_should_return_o_series_config_when_base_model_is_o_series(self):
        config = ProviderConfigManager._get_azure_config(
            model="my-deployment-id", base_model="azure/o4-mini"
        )
        assert isinstance(config, AzureOpenAIO1Config)

    def test_should_return_default_config_when_base_model_is_regular(self):
        config = ProviderConfigManager._get_azure_config(
            model="my-deployment-id", base_model="azure/gpt-4o"
        )
        assert type(config).__name__ == "AzureOpenAIConfig"

    def test_should_fallback_to_model_when_base_model_is_none(self):
        config = ProviderConfigManager._get_azure_config(
            model="gpt-5.2", base_model=None
        )
        assert isinstance(config, AzureOpenAIGPT5Config)

    def test_should_return_default_config_when_both_are_non_standard(self):
        config = ProviderConfigManager._get_azure_config(
            model="my-deployment-id", base_model=None
        )
        assert type(config).__name__ == "AzureOpenAIConfig"


# ---------------------------------------------------------------------------
# get_provider_chat_config — threads base_model through for Azure
# ---------------------------------------------------------------------------
class TestGetProviderChatConfigWithBaseModel:
    """get_provider_chat_config should pass base_model to Azure config selection."""

    def test_should_return_gpt5_config_for_custom_deployment_with_base_model(self):
        from litellm.types.utils import LlmProviders

        config = ProviderConfigManager.get_provider_chat_config(
            model="my-deployment-id",
            provider=LlmProviders.AZURE,
            base_model="azure/gpt-5",
        )
        assert isinstance(config, AzureOpenAIGPT5Config)

    def test_should_return_o_series_config_for_custom_deployment_with_base_model(self):
        from litellm.types.utils import LlmProviders

        config = ProviderConfigManager.get_provider_chat_config(
            model="my-other-deployment",
            provider=LlmProviders.AZURE,
            base_model="azure/o3-mini",
        )
        assert isinstance(config, AzureOpenAIO1Config)


# ---------------------------------------------------------------------------
# get_supported_openai_params — base_model drives Azure param detection
# ---------------------------------------------------------------------------
class TestGetSupportedOpenAIParamsWithBaseModel:
    """get_supported_openai_params should use base_model for Azure detection."""

    def test_should_return_gpt5_params_for_custom_deployment_with_gpt5_base_model(
        self,
    ):
        params = litellm.get_supported_openai_params(
            model="my-deployment-id",
            custom_llm_provider="azure",
            base_model="azure/gpt-5",
        )
        assert params is not None
        assert "reasoning_effort" in params
        # gpt-5 maps max_tokens -> max_completion_tokens, verifying we got GPT-5 config
        assert "max_completion_tokens" in params

    def test_should_return_o_series_params_for_custom_deployment_with_o_series_base_model(
        self,
    ):
        params = litellm.get_supported_openai_params(
            model="my-other-deployment",
            custom_llm_provider="azure",
            base_model="azure/o4-mini",
        )
        assert params is not None
        assert "reasoning_effort" in params

    def test_should_return_regular_params_when_no_base_model(self):
        """When base_model is not set and model is non-standard, default Azure config."""
        params = litellm.get_supported_openai_params(
            model="my-deployment-id",
            custom_llm_provider="azure",
        )
        assert params is not None
        # Default Azure config supports temperature
        assert "temperature" in params


# ---------------------------------------------------------------------------
# get_optional_params — base_model drives Azure param mapping
# ---------------------------------------------------------------------------
class TestGetOptionalParamsWithBaseModel:
    """get_optional_params should use base_model for Azure model-type detection."""

    def test_should_map_max_tokens_for_custom_deployment_with_gpt5_base_model(self):
        """A non-standard deployment name + gpt-5 base_model should map max_tokens -> max_completion_tokens."""
        params = get_optional_params(
            model="my-deployment-id",
            custom_llm_provider="azure",
            max_tokens=100,
            base_model="azure/gpt-5",
        )
        assert params.get("max_completion_tokens") == 100
        assert "max_tokens" not in params

    def test_should_keep_max_tokens_for_custom_deployment_without_base_model(self):
        """A non-standard deployment name without base_model should use default Azure config."""
        params = get_optional_params(
            model="my-deployment-id",
            custom_llm_provider="azure",
            max_tokens=100,
            api_version="2024-05-01-preview",
        )
        # Default AzureOpenAIConfig keeps max_tokens as-is (or maps based on api_version)
        assert "max_tokens" in params or "max_completion_tokens" in params

    def test_should_support_reasoning_effort_for_custom_deployment_with_o_series_base_model(
        self,
    ):
        """A non-standard deployment name + o-series base_model should accept reasoning_effort."""
        params = get_optional_params(
            model="my-other-deployment",
            custom_llm_provider="azure",
            reasoning_effort="low",
            base_model="azure/o4-mini",
        )
        assert params.get("reasoning_effort") == "low"

    def test_should_reject_temperature_for_custom_deployment_with_gpt5_base_model(
        self,
    ):
        """A non-standard deployment + gpt-5 base_model should reject temperature."""
        with pytest.raises(litellm.UnsupportedParamsError):
            get_optional_params(
                model="my-deployment-id",
                custom_llm_provider="azure",
                temperature=0.5,
                base_model="azure/gpt-5",
            )


# ---------------------------------------------------------------------------
# Backward compatibility — existing patterns still work
# ---------------------------------------------------------------------------
class TestBackwardCompatibility:
    """Existing model-name-based and prefix-based patterns must keep working."""

    def test_should_detect_gpt5_from_model_name(self):
        config = ProviderConfigManager._get_azure_config(model="gpt-5.2")
        assert isinstance(config, AzureOpenAIGPT5Config)

    def test_should_detect_gpt5_from_gpt5_series_prefix(self):
        config = ProviderConfigManager._get_azure_config(
            model="gpt5_series/my-deployment"
        )
        assert isinstance(config, AzureOpenAIGPT5Config)

    def test_should_detect_o_series_from_model_name(self):
        config = ProviderConfigManager._get_azure_config(model="o4-mini")
        assert isinstance(config, AzureOpenAIO1Config)

    def test_should_detect_o_series_from_o_series_prefix(self):
        config = ProviderConfigManager._get_azure_config(model="o_series/my-deployment")
        assert isinstance(config, AzureOpenAIO1Config)

    def test_should_handle_gpt5_chat_model_correctly(self):
        """gpt-5-chat models should NOT be routed to GPT-5 config."""
        config = ProviderConfigManager._get_azure_config(model="gpt-5-chat")
        assert type(config).__name__ == "AzureOpenAIConfig"

    def test_base_model_overrides_model_detection(self):
        """base_model should take priority over model for type detection."""
        # model looks like o-series, but base_model says gpt-5
        config = ProviderConfigManager._get_azure_config(
            model="o3-mini", base_model="azure/gpt-5.2"
        )
        assert isinstance(config, AzureOpenAIGPT5Config)


# ---------------------------------------------------------------------------
# Deep config method awareness — base_model flows into config internals
# ---------------------------------------------------------------------------
class TestBaseModelFlowsIntoConfigInternals:
    """base_model should be used by config internal methods (e.g. is_model_gpt_5_2_model)."""

    def test_should_support_logprobs_for_prefixed_deployment_with_gpt52_base_model(
        self,
    ):
        """Deployment 'my-gpt-5.2' with base_model='azure/gpt-5.2' should support logprobs."""
        params = litellm.get_supported_openai_params(
            model="gpt5_series/my-gpt-5.2",
            custom_llm_provider="azure",
            base_model="azure/gpt-5.2",
        )
        assert params is not None
        assert "logprobs" in params
        assert "top_logprobs" in params

    def test_should_support_logprobs_for_plain_deployment_with_gpt52_base_model(self):
        """Deployment 'my-deployment-id' with base_model='azure/gpt-5.2' should support logprobs."""
        params = litellm.get_supported_openai_params(
            model="my-deployment-id",
            custom_llm_provider="azure",
            base_model="azure/gpt-5.2",
        )
        assert params is not None
        assert "logprobs" in params
        assert "top_logprobs" in params

    def test_should_not_support_logprobs_for_gpt5_base_model(self):
        """Deployment with base_model='azure/gpt-5' (not 5.2) should NOT support logprobs."""
        params = litellm.get_supported_openai_params(
            model="my-deployment-id",
            custom_llm_provider="azure",
            base_model="azure/gpt-5",
        )
        assert params is not None
        assert "logprobs" not in params
        assert "top_logprobs" not in params

    def test_should_pass_logprobs_through_get_optional_params(self):
        """logprobs should pass validation in get_optional_params when base_model is gpt-5.2."""
        params = get_optional_params(
            model="gpt5_series/my-gpt-5.2",
            custom_llm_provider="azure",
            logprobs=True,
            top_logprobs=5,
            base_model="azure/gpt-5.2",
        )
        assert params.get("logprobs") is True
        assert params.get("top_logprobs") == 5

    def test_should_map_max_tokens_for_prefixed_deployment_with_gpt5_base_model(self):
        """my-gpt-5.2 with base_model should correctly map max_tokens -> max_completion_tokens."""
        params = get_optional_params(
            model="gpt5_series/my-gpt-5.2",
            custom_llm_provider="azure",
            max_tokens=200,
            base_model="azure/gpt-5.2",
        )
        assert params.get("max_completion_tokens") == 200
        assert "max_tokens" not in params
