import pytest

import litellm
from litellm.llms.azure.chat.gpt_5_transformation import AzureOpenAIGPT5Config


@pytest.fixture()
def config() -> AzureOpenAIGPT5Config:
    return AzureOpenAIGPT5Config()


def test_azure_gpt5_supports_reasoning_effort(config: AzureOpenAIGPT5Config):
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt-5")
    assert "reasoning_effort" in config.get_supported_openai_params(
        model="gpt5_series/my-deployment"
    )


def test_azure_gpt5_allows_tool_choice_for_deployment_names():
    supported_params = litellm.get_supported_openai_params(
        model="gpt-5-chat-2025-08-07", custom_llm_provider="azure"
    )
    assert supported_params is not None
    assert "tool_choice" in supported_params
    # gpt-5-chat* should not be treated as a GPT-5 reasoning model
    assert "reasoning_effort" not in supported_params
    assert "temperature" in supported_params


def test_azure_gpt5_maps_max_tokens(config: AzureOpenAIGPT5Config):
    params = config.map_openai_params(
        non_default_params={"max_tokens": 5},
        optional_params={},
        model="gpt5_series/gpt-5",
        drop_params=False,
        api_version="2024-05-01-preview",
    )
    assert params["max_completion_tokens"] == 5
    assert "max_tokens" not in params


def test_azure_gpt5_temperature_error(config: AzureOpenAIGPT5Config):
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"temperature": 0.2},
            optional_params={},
            model="gpt-5",
            drop_params=False,
            api_version="2024-05-01-preview",
        )


def test_azure_gpt5_series_transform_request(config: AzureOpenAIGPT5Config):
    request = config.transform_request(
        model="gpt5_series/gpt-5",
        messages=[],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert request["model"] == "gpt-5"


# GPT-5-Codex specific tests for Azure
def test_azure_gpt5_codex_model_detection(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5-Codex models are correctly detected."""
    assert config.is_model_gpt_5_model("gpt-5-codex")
    assert config.is_model_gpt_5_model("gpt5_series/gpt-5-codex")


def test_azure_gpt5_codex_supports_reasoning_effort(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5-Codex supports reasoning_effort parameter."""
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt-5-codex")
    assert "reasoning_effort" in config.get_supported_openai_params(
        model="gpt5_series/gpt-5-codex"
    )


def test_azure_gpt5_codex_maps_max_tokens(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5-Codex correctly maps max_tokens to max_completion_tokens."""
    params = config.map_openai_params(
        non_default_params={"max_tokens": 150},
        optional_params={},
        model="gpt-5-codex",
        drop_params=False,
        api_version="2024-05-01-preview",
    )
    assert params["max_completion_tokens"] == 150
    assert "max_tokens" not in params


def test_azure_gpt5_codex_temperature_error(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5-Codex raises error for unsupported temperature."""
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"temperature": 0.8},
            optional_params={},
            model="gpt-5-codex",
            drop_params=False,
            api_version="2024-05-01-preview",
        )


def test_azure_gpt5_codex_series_transform_request(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5-Codex series routing works correctly."""
    request = config.transform_request(
        model="gpt5_series/gpt-5-codex",
        messages=[],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert request["model"] == "gpt-5-codex"


# GPT-5.1 temperature handling tests for Azure
def test_azure_gpt5_1_temperature_with_reasoning_effort_none(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5.1 supports any temperature when reasoning_effort='none'.

    Azure OpenAI supports reasoning_effort='none' for gpt-5.1 models.
    See: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning
    """
    params = config.map_openai_params(
        non_default_params={"temperature": 0.5, "reasoning_effort": "none"},
        optional_params={},
        model="azure/gpt-5.1",
        drop_params=False,
        api_version="2024-05-01-preview",
    )
    assert params["temperature"] == 0.5
    # Azure supports reasoning_effort="none" for gpt-5.1
    assert params.get("reasoning_effort") == "none"


def test_azure_gpt5_1_reasoning_effort_none_supported(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5.1 supports reasoning_effort='none' without error."""
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "none"},
        optional_params={},
        model="azure/gpt-5.1",
        drop_params=False,
        api_version="2024-05-01-preview",
    )
    assert params.get("reasoning_effort") == "none"


def test_azure_gpt5_1_temperature_without_reasoning_effort(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5.1 supports any temperature when reasoning_effort is not specified."""
    params = config.map_openai_params(
        non_default_params={"temperature": 0.7},
        optional_params={},
        model="azure/gpt-5.1",
        drop_params=False,
        api_version="2024-05-01-preview",
    )
    assert params["temperature"] == 0.7


def test_azure_gpt5_1_temperature_with_reasoning_effort_other_values(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5.1 only allows temperature=1 when reasoning_effort is not 'none'."""
    # Test that temperature != 1 raises error when reasoning_effort is set to other values
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"temperature": 0.7, "reasoning_effort": "low"},
            optional_params={},
            model="azure/gpt-5.1",
            drop_params=False,
            api_version="2024-05-01-preview",
        )
    
    # Test that temperature=1 is allowed with other reasoning_effort values
    params = config.map_openai_params(
        non_default_params={"temperature": 1.0, "reasoning_effort": "medium"},
        optional_params={},
        model="azure/gpt-5.1",
        drop_params=False,
        api_version="2024-05-01-preview",
    )
    assert params["temperature"] == 1.0
    assert params["reasoning_effort"] == "medium"


def test_azure_gpt5_1_series_temperature_handling(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5.1 with gpt5_series prefix supports temperature with reasoning_effort='none'."""
    params = config.map_openai_params(
        non_default_params={"temperature": 0.6},
        optional_params={},
        model="gpt5_series/gpt-5.1",
        drop_params=False,
        api_version="2024-05-01-preview",
    )
    assert params["temperature"] == 0.6


def test_azure_gpt5_reasoning_effort_none_error(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5 (non-5.1) raises error for reasoning_effort='none' when drop_params=False."""
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={},
            model="azure/gpt-5",
            drop_params=False,
            api_version="2024-05-01-preview",
        )


def test_azure_gpt5_reasoning_effort_none_dropped(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5 (non-5.1) drops reasoning_effort='none' when drop_params=True."""
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "none"},
        optional_params={},
        model="azure/gpt-5",
        drop_params=True,
        api_version="2024-05-01-preview",
    )
    assert "reasoning_effort" not in params or params.get("reasoning_effort") != "none"


# Logprobs support tests for Azure GPT-5.2
def test_azure_gpt5_2_supports_logprobs(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5.2 models support logprobs parameters.

    Only Azure OpenAI GPT-5.2 supports logprobs, unlike OpenAI's GPT-5 or Azure's gpt-5/gpt-5.1.
    Tested with gpt-5.2 on api-version 2025-01-01-preview.
    """
    supported_params = config.get_supported_openai_params(model="gpt-5.2")
    assert "logprobs" in supported_params
    assert "top_logprobs" in supported_params


def test_azure_gpt5_2_with_prefix_supports_logprobs(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5.2 with azure/ prefix supports logprobs parameters."""
    supported_params = config.get_supported_openai_params(model="azure/gpt-5.2")
    assert "logprobs" in supported_params
    assert "top_logprobs" in supported_params


def test_azure_gpt5_2_series_supports_logprobs(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5.2 with gpt5_series prefix supports logprobs."""
    supported_params = config.get_supported_openai_params(model="gpt5_series/gpt-5.2")
    assert "logprobs" in supported_params
    assert "top_logprobs" in supported_params


def test_azure_gpt5_2_logprobs_params_passed_through(config: AzureOpenAIGPT5Config):
    """Test that logprobs parameters are correctly passed through to the API for gpt-5.2."""
    params = config.map_openai_params(
        non_default_params={"logprobs": True, "top_logprobs": 5},
        optional_params={},
        model="azure/gpt-5.2",
        drop_params=False,
        api_version="2025-01-01-preview",
    )
    assert params["logprobs"] is True
    assert params["top_logprobs"] == 5


def test_azure_gpt5_base_does_not_support_logprobs(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5 (non-5.2) does not support logprobs parameters.

    Only gpt-5.2 has been verified to support logprobs on Azure.
    """
    supported_params = config.get_supported_openai_params(model="gpt-5")
    assert "logprobs" not in supported_params
    assert "top_logprobs" not in supported_params


def test_azure_gpt5_1_does_not_support_logprobs(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5.1 does not support logprobs parameters.

    Only gpt-5.2 has been verified to support logprobs on Azure.
    """
    supported_params = config.get_supported_openai_params(model="gpt-5.1")
    assert "logprobs" not in supported_params
    assert "top_logprobs" not in supported_params

