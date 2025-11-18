"""Test Anthropic thinking parameter conversion to OpenAI reasoning_effort."""

import pytest

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config
from litellm.llms.openai.chat.o_series_transformation import OpenAIOSeriesConfig


@pytest.fixture()
def gpt_config() -> OpenAIGPTConfig:
    return OpenAIGPTConfig()


@pytest.fixture()
def gpt5_config() -> OpenAIGPT5Config:
    return OpenAIGPT5Config()


@pytest.fixture()
def o_series_config() -> OpenAIOSeriesConfig:
    return OpenAIOSeriesConfig()


# Test the static method for thinking -> reasoning_effort conversion
def test_map_thinking_to_reasoning_effort_minimal():
    """Test that low budget_tokens maps to 'minimal' reasoning_effort."""
    result = OpenAIGPTConfig._map_thinking_to_reasoning_effort(
        {"type": "enabled", "budget_tokens": 128}
    )
    assert result == "minimal"


def test_map_thinking_to_reasoning_effort_low():
    """Test that medium budget_tokens maps to 'low' reasoning_effort."""
    result = OpenAIGPTConfig._map_thinking_to_reasoning_effort(
        {"type": "enabled", "budget_tokens": 1024}
    )
    assert result == "low"


def test_map_thinking_to_reasoning_effort_medium():
    """Test that higher budget_tokens maps to 'medium' reasoning_effort."""
    result = OpenAIGPTConfig._map_thinking_to_reasoning_effort(
        {"type": "enabled", "budget_tokens": 2048}
    )
    assert result == "medium"


def test_map_thinking_to_reasoning_effort_high():
    """Test that high budget_tokens maps to 'high' reasoning_effort."""
    result = OpenAIGPTConfig._map_thinking_to_reasoning_effort(
        {"type": "enabled", "budget_tokens": 4096}
    )
    assert result == "high"


def test_map_thinking_to_reasoning_effort_no_budget():
    """Test that enabled thinking without budget_tokens defaults to 'medium'."""
    result = OpenAIGPTConfig._map_thinking_to_reasoning_effort({"type": "enabled"})
    assert result == "medium"


def test_map_thinking_to_reasoning_effort_disabled():
    """Test that disabled thinking returns None."""
    result = OpenAIGPTConfig._map_thinking_to_reasoning_effort(
        {"type": "disabled", "budget_tokens": 1024}
    )
    assert result is None


def test_map_thinking_to_reasoning_effort_none():
    """Test that None thinking returns None."""
    result = OpenAIGPTConfig._map_thinking_to_reasoning_effort(None)
    assert result is None


# Test the static method for reasoning_content -> thinking_blocks conversion
def test_convert_reasoning_content_to_thinking_blocks():
    """Test that reasoning_content is properly converted to thinking_blocks format."""
    reasoning_text = "Let me think about this step by step..."
    result = OpenAIGPTConfig._convert_reasoning_content_to_thinking_blocks(
        reasoning_text
    )

    assert result is not None
    assert len(result) == 1
    assert result[0]["type"] == "thinking"
    assert result[0]["thinking"] == reasoning_text
    assert result[0]["signature"] == ""


def test_convert_reasoning_content_to_thinking_blocks_none():
    """Test that None reasoning_content returns None."""
    result = OpenAIGPTConfig._convert_reasoning_content_to_thinking_blocks(None)
    assert result is None


def test_convert_reasoning_content_to_thinking_blocks_empty():
    """Test that empty reasoning_content returns None."""
    result = OpenAIGPTConfig._convert_reasoning_content_to_thinking_blocks("")
    assert result is None


# Integration tests: thinking parameter in map_openai_params
def test_gpt5_thinking_param_conversion(gpt5_config: OpenAIGPT5Config):
    """Test that thinking parameter is converted to reasoning_effort for GPT-5."""
    params = gpt5_config.map_openai_params(
        non_default_params={"thinking": {"type": "enabled", "budget_tokens": 1024}},
        optional_params={},
        model="gpt-5",
        drop_params=False,
    )

    # Should have reasoning_effort instead of thinking
    assert params["reasoning_effort"] == "low"
    assert "_original_thinking_param" in params
    assert params["_original_thinking_param"] == {
        "type": "enabled",
        "budget_tokens": 1024,
    }


def test_o_series_thinking_param_conversion(o_series_config: OpenAIOSeriesConfig):
    """Test that thinking parameter is converted to reasoning_effort for O-series."""
    params = o_series_config.map_openai_params(
        non_default_params={"thinking": {"type": "enabled", "budget_tokens": 2048}},
        optional_params={},
        model="o1",
        drop_params=False,
    )

    # Should have reasoning_effort instead of thinking
    assert params["reasoning_effort"] == "medium"
    assert "_original_thinking_param" in params


def test_thinking_param_not_converted_for_non_reasoning_models(
    gpt_config: OpenAIGPTConfig,
):
    """Test that thinking parameter is skipped for models without reasoning_effort support."""
    params = gpt_config.map_openai_params(
        non_default_params={"thinking": {"type": "enabled", "budget_tokens": 1024}},
        optional_params={},
        model="gpt-4",  # GPT-4 doesn't support reasoning_effort
        drop_params=False,
    )

    # Should not have reasoning_effort
    assert "reasoning_effort" not in params
    # Should not store original thinking param since conversion didn't happen
    assert "_original_thinking_param" not in params


def test_thinking_disabled_no_conversion(gpt5_config: OpenAIGPT5Config):
    """Test that disabled thinking parameter doesn't add reasoning_effort."""
    params = gpt5_config.map_openai_params(
        non_default_params={"thinking": {"type": "disabled"}},
        optional_params={},
        model="gpt-5",
        drop_params=False,
    )

    # Should not have reasoning_effort
    assert "reasoning_effort" not in params

