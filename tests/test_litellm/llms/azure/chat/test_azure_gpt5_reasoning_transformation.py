import pytest

import litellm
from litellm.llms.azure.chat.gpt_5_reasoning_transformation import AzureOpenAIGPT5ReasoningConfig


@pytest.fixture()
def config() -> AzureOpenAIGPT5ReasoningConfig:
    return AzureOpenAIGPT5ReasoningConfig()


def test_azure_gpt5_supports_reasoning_effort(config: AzureOpenAIGPT5ReasoningConfig):
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt-5")
    assert "reasoning_effort" in config.get_supported_openai_params(
        model="gpt5_series/my-deployment"
    )

def test_azure_gpt5_maps_max_tokens(config: AzureOpenAIGPT5ReasoningConfig):
    params = config.map_openai_params(
        non_default_params={"max_tokens": 5},
        optional_params={},
        model="gpt5_series/gpt-5",
        drop_params=False,
        api_version="2024-05-01-preview",
    )
    assert params["max_completion_tokens"] == 5
    assert "max_tokens" not in params


def test_azure_gpt5_temperature_error(config: AzureOpenAIGPT5ReasoningConfig):
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"temperature": 0.2},
            optional_params={},
            model="gpt-5",
            drop_params=False,
            api_version="2024-05-01-preview",
        )

def test_azure_gpt5_series_transform_request(config: AzureOpenAIGPT5ReasoningConfig):
    request = config.transform_request(
        model="gpt5_series/gpt-5",
        messages=[],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert request["model"] == "gpt-5"


# GPT-5-Codex specific tests for Azure
def test_azure_gpt5_codex_model_detection(config: AzureOpenAIGPT5ReasoningConfig):
    """Test that Azure GPT-5-Codex models are correctly detected."""
    assert config.is_model_gpt_5_reasoning_model("gpt-5-codex")
    assert config.is_model_gpt_5_reasoning_model("gpt5_series/gpt-5-codex")


def test_azure_gpt5_codex_supports_reasoning_effort(config: AzureOpenAIGPT5ReasoningConfig):
    """Test that Azure GPT-5-Codex supports reasoning_effort parameter."""
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt-5-codex")
    assert "reasoning_effort" in config.get_supported_openai_params(
        model="gpt5_series/gpt-5-codex"
    )


def test_azure_gpt5_codex_maps_max_tokens(config: AzureOpenAIGPT5ReasoningConfig):
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


def test_azure_gpt5_codex_temperature_error(config: AzureOpenAIGPT5ReasoningConfig):
    """Test that Azure GPT-5-Codex raises error for unsupported temperature."""
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"temperature": 0.8},
            optional_params={},
            model="gpt-5-codex",
            drop_params=False,
            api_version="2024-05-01-preview",
        )


def test_azure_gpt5_codex_series_transform_request(config: AzureOpenAIGPT5ReasoningConfig):
    """Test that Azure GPT-5-Codex series routing works correctly."""
    request = config.transform_request(
        model="gpt5_series/gpt-5-codex",
        messages=[],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert request["model"] == "gpt-5-codex"

