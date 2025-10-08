import pytest

import litellm
from litellm.llms.openai.openai import OpenAIConfig
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config


@pytest.fixture()
def config() -> OpenAIConfig:
    return OpenAIConfig()


@pytest.fixture()
def gpt5_config() -> OpenAIGPT5Config:
    return OpenAIGPT5Config()


def test_gpt5_supports_reasoning_effort(config: OpenAIConfig):
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt-5")
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt-5-mini")


def test_gpt5_maps_max_tokens(config: OpenAIConfig):
    params = config.map_openai_params(
        non_default_params={"max_tokens": 10},
        optional_params={},
        model="gpt-5",
        drop_params=False,
    )
    assert params["max_completion_tokens"] == 10
    assert "max_tokens" not in params


def test_gpt5_temperature_drop(config: OpenAIConfig):
    params = config.map_openai_params(
        non_default_params={"temperature": 0.2},
        optional_params={},
        model="gpt-5",
        drop_params=True,
    )
    assert "temperature" not in params


def test_gpt5_temperature_error(config: OpenAIConfig):
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"temperature": 0.2},
            optional_params={},
            model="gpt-5",
            drop_params=False,
        )


def test_gpt5_unsupported_params_drop(config: OpenAIConfig):
    assert "top_p" not in config.get_supported_openai_params(model="gpt-5")
    params = config.map_openai_params(
        non_default_params={"top_p": 0.5},
        optional_params={},
        model="gpt-5",
        drop_params=True,
    )
    assert "top_p" not in params


# GPT-5-Codex specific tests
def test_gpt5_codex_model_detection(gpt5_config: OpenAIGPT5Config):
    """Test that GPT-5-Codex models are correctly detected as GPT-5 models."""
    assert gpt5_config.is_model_gpt_5_model("gpt-5-codex")
    assert gpt5_config.is_model_gpt_5_codex_model("gpt-5-codex")

    # Regular GPT-5 models should not be detected as codex
    assert not gpt5_config.is_model_gpt_5_codex_model("gpt-5")
    assert not gpt5_config.is_model_gpt_5_codex_model("gpt-5-mini")


def test_gpt5_codex_supports_reasoning_effort(config: OpenAIConfig):
    """Test that GPT-5-Codex supports reasoning_effort parameter."""
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt-5-codex")


def test_gpt5_codex_maps_max_tokens(config: OpenAIConfig):
    """Test that GPT-5-Codex correctly maps max_tokens to max_completion_tokens."""
    params = config.map_openai_params(
        non_default_params={"max_tokens": 100},
        optional_params={},
        model="gpt-5-codex",
        drop_params=False,
    )
    assert params["max_completion_tokens"] == 100
    assert "max_tokens" not in params


def test_gpt5_codex_temperature_drop(config: OpenAIConfig):
    """Test that GPT-5-Codex drops unsupported temperature values when drop_params=True."""
    params = config.map_openai_params(
        non_default_params={"temperature": 0.7},
        optional_params={},
        model="gpt-5-codex",
        drop_params=True,
    )
    assert "temperature" not in params


def test_gpt5_codex_temperature_error(config: OpenAIConfig):
    """Test that GPT-5-Codex raises error for unsupported temperature when drop_params=False."""
    with pytest.raises(
        litellm.utils.UnsupportedParamsError,
        match="gpt-5 models \\(including gpt-5-codex\\)",
    ):
        config.map_openai_params(
            non_default_params={"temperature": 0.7},
            optional_params={},
            model="gpt-5-codex",
            drop_params=False,
        )



def test_gpt5_codex_temperature_one_allowed(config: OpenAIConfig):
    """Test that GPT-5-Codex allows temperature=1."""
    params = config.map_openai_params(
        non_default_params={"temperature": 1.0},
        optional_params={},
        model="gpt-5-codex",
        drop_params=False,
    )
    assert params["temperature"] == 1.0


def test_gpt5_codex_unsupported_params_drop(config: OpenAIConfig):
    """Test that GPT-5-Codex drops unsupported parameters."""
    unsupported_params = [
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "top_logprobs",
    ]

    for param in unsupported_params:
        assert param not in config.get_supported_openai_params(model="gpt-5-codex")


def test_gpt5_codex_supports_tool_choice(gpt5_config: OpenAIGPT5Config):
    """Test that GPT-5-Codex supports tool_choice parameter."""
    supported_params = gpt5_config.get_supported_openai_params(model="gpt-5-codex")
    assert "tool_choice" in supported_params


def test_gpt5_codex_supports_function_calling(config: OpenAIConfig):
    """Test that GPT-5-Codex supports function calling parameters."""
    supported_params = config.get_supported_openai_params(model="gpt-5-codex")
    assert "functions" in supported_params
    assert "function_call" in supported_params
    assert "tools" in supported_params
