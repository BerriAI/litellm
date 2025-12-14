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


def test_gpt5_verbosity_parameter(config: OpenAIConfig):
    """Test that verbosity parameter passes through correctly for GPT-5 models.

    The verbosity parameter controls output length and detail for GPT-5 family models.
    Supported values: 'low', 'medium', 'high'
    Supported models: gpt-5, gpt-5.1, gpt-5-mini, gpt-5-nano, gpt-5-codex, gpt-5-pro
    """
    # Test all valid verbosity values
    for verbosity_level in ["low", "medium", "high"]:
        params = config.map_openai_params(
            non_default_params={"verbosity": verbosity_level},
            optional_params={},
            model="gpt-5.1",
            drop_params=False,
        )
        assert params["verbosity"] == verbosity_level

    # Test with different GPT-5 models
    for model in ["gpt-5", "gpt-5.1", "gpt-5-mini", "gpt-5-nano", "gpt-5-codex"]:
        params = config.map_openai_params(
            non_default_params={"verbosity": "low"},
            optional_params={},
            model=model,
            drop_params=False,
        )
        assert params["verbosity"] == "low"
def test_gpt5_1_reasoning_effort_none(config: OpenAIConfig):
    """Test that GPT-5.1 supports reasoning_effort='none' parameter.

    Related issue: https://github.com/BerriAI/litellm/issues/16633
    GPT-5.1 introduced 'none' as the new default reasoning effort setting
    for faster, lower-latency responses.
    """
    # Test that reasoning_effort is a supported parameter
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt-5.1")

    # Test that reasoning_effort="none" passes through correctly
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "none"},
        optional_params={},
        model="gpt-5.1",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "none"

    # Test with other valid values for GPT-5.1
    for effort in ["low", "medium", "high"]:
        params = config.map_openai_params(
            non_default_params={"reasoning_effort": effort},
            optional_params={},
            model="gpt-5.1",
            drop_params=False,
        )
        assert params["reasoning_effort"] == effort


def test_gpt5_1_codex_max_allows_reasoning_effort_xhigh(config: OpenAIConfig):
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "xhigh"},
        optional_params={},
        model="gpt-5.1-codex-max",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "xhigh"


def test_gpt5_rejects_reasoning_effort_xhigh_for_other_models(config: OpenAIConfig):
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"reasoning_effort": "xhigh"},
            optional_params={},
            model="gpt-5.1",
            drop_params=False,
        )


def test_gpt5_drops_reasoning_effort_xhigh_when_requested(config: OpenAIConfig):
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "xhigh"},
        optional_params={},
        model="gpt-5",
        drop_params=True,
    )
    assert "reasoning_effort" not in params


# GPT-5.1 temperature handling tests
def test_gpt5_1_model_detection(gpt5_config: OpenAIGPT5Config):
    """Test that GPT-5.1 models are correctly detected."""
    assert gpt5_config.is_model_gpt_5_1_model("gpt-5.1")
    assert gpt5_config.is_model_gpt_5_1_model("gpt-5.1-codex")
    assert gpt5_config.is_model_gpt_5_1_model("gpt-5.1-codex-max")
    assert gpt5_config.is_model_gpt_5_1_model("gpt-5.1-chat")
    assert gpt5_config.is_model_gpt_5_1_model("gpt-5.2")
    assert gpt5_config.is_model_gpt_5_1_model("gpt-5.2-2025-12-11")
    assert gpt5_config.is_model_gpt_5_1_model("gpt-5.2-chat-latest")
    assert not gpt5_config.is_model_gpt_5_1_model("gpt-5.2-pro")
    assert not gpt5_config.is_model_gpt_5_1_model("gpt-5")
    assert not gpt5_config.is_model_gpt_5_1_model("gpt-5-mini")
    assert not gpt5_config.is_model_gpt_5_1_model("gpt-5-codex")


def test_gpt5_1_temperature_with_reasoning_effort_none(config: OpenAIConfig):
    """Test that GPT-5.1 supports any temperature when reasoning_effort='none'."""
    # Test various temperature values with reasoning_effort="none"
    for temp in [0.0, 0.2, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]:
        params = config.map_openai_params(
            non_default_params={"temperature": temp, "reasoning_effort": "none"},
            optional_params={},
            model="gpt-5.1",
            drop_params=False,
        )
        assert params["temperature"] == temp
        assert params["reasoning_effort"] == "none"


def test_gpt5_2_temperature_with_reasoning_effort_none(config: OpenAIConfig):
    """Test that GPT-5.2 aligns with GPT-5.1 temperature rules when effort='none'."""
    for temp in [0.0, 0.3, 0.7, 1.0, 1.5]:
        params = config.map_openai_params(
            non_default_params={"temperature": temp, "reasoning_effort": "none"},
            optional_params={},
            model="gpt-5.2",
            drop_params=False,
        )
        assert params["temperature"] == temp
        assert params["reasoning_effort"] == "none"


def test_gpt5_1_temperature_without_reasoning_effort(config: OpenAIConfig):
    """Test that GPT-5.1 supports any temperature when reasoning_effort is not specified.
    
    When reasoning_effort is not provided, it defaults to "none" for gpt-5.1,
    so temperature should be allowed.
    """
    # Test various temperature values without reasoning_effort (defaults to "none")
    for temp in [0.0, 0.2, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]:
        params = config.map_openai_params(
            non_default_params={"temperature": temp},
            optional_params={},
            model="gpt-5.1",
            drop_params=False,
        )
        assert params["temperature"] == temp


def test_gpt5_1_temperature_with_reasoning_effort_other_values(config: OpenAIConfig):
    """Test that GPT-5.1 only allows temperature=1 when reasoning_effort is not 'none'."""
    # Test that temperature != 1 raises error when reasoning_effort is set to other values
    for effort in ["low", "medium", "high"]:
        with pytest.raises(litellm.utils.UnsupportedParamsError):
            config.map_openai_params(
                non_default_params={"temperature": 0.7, "reasoning_effort": effort},
                optional_params={},
                model="gpt-5.1",
                drop_params=False,
            )
    
    # Test that temperature=1 is allowed with other reasoning_effort values
    for effort in ["low", "medium", "high"]:
        params = config.map_openai_params(
            non_default_params={"temperature": 1.0, "reasoning_effort": effort},
            optional_params={},
            model="gpt-5.1",
            drop_params=False,
        )
        assert params["temperature"] == 1.0
        assert params["reasoning_effort"] == effort


def test_gpt5_1_temperature_with_reasoning_effort_in_optional_params(config: OpenAIConfig):
    """Test that reasoning_effort can be in optional_params and still work correctly."""
    # Test with reasoning_effort="none" in optional_params
    params = config.map_openai_params(
        non_default_params={"temperature": 0.5},
        optional_params={"reasoning_effort": "none"},
        model="gpt-5.1",
        drop_params=False,
    )
    assert params["temperature"] == 0.5
    
    # Test with reasoning_effort="low" in optional_params (should only allow temp=1)
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"temperature": 0.5},
            optional_params={"reasoning_effort": "low"},
            model="gpt-5.1",
            drop_params=False,
        )

def test_gpt5_1_temperature_drop_when_not_none(config: OpenAIConfig):
    """Test that GPT-5.1 drops temperature when reasoning_effort != 'none' and drop_params=True."""
    params = config.map_openai_params(
        non_default_params={"temperature": 0.7, "reasoning_effort": "low"},
        optional_params={},
        model="gpt-5.1",
        drop_params=True,
    )
    assert "temperature" not in params
    assert params["reasoning_effort"] == "low"


def test_gpt5_temperature_still_restricted(config: OpenAIConfig):
    """Test that regular gpt-5 (not 5.1) still only allows temperature=1."""
    # Regular gpt-5 should still only allow temperature=1
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"temperature": 0.7},
            optional_params={},
            model="gpt-5",
            drop_params=False,
        )
    
    # temperature=1 should still work for gpt-5
    params = config.map_openai_params(
        non_default_params={"temperature": 1.0},
        optional_params={},
        model="gpt-5",
        drop_params=False,
    )
    assert params["temperature"] == 1.0


def test_gpt5_2_pro_allows_reasoning_effort_xhigh(config: OpenAIConfig):
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "xhigh"},
        optional_params={},
        model="gpt-5.2-pro",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "xhigh"


def test_gpt5_2_allows_reasoning_effort_xhigh(config: OpenAIConfig):
    """Test that gpt-5.2 (base model) also supports reasoning_effort='xhigh'."""
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "xhigh"},
        optional_params={},
        model="gpt-5.2",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "xhigh"
