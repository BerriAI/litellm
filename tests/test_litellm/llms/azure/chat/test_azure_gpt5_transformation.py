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

