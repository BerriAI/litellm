import pytest

import litellm
import litellm.main as litellm_main
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config
from litellm.llms.openai.openai import OpenAIConfig
from litellm.utils import (
    _is_explicitly_disabled_factory,
    peek_reasoning_summary_aliases,
    strip_reasoning_summary_aliases_from_optional_params,
)


@pytest.fixture()
def config() -> OpenAIConfig:
    return OpenAIConfig()


@pytest.fixture()
def gpt5_config() -> OpenAIGPT5Config:
    return OpenAIGPT5Config()


@pytest.fixture(autouse=True)
def use_local_model_cost_map(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        litellm, "model_cost", get_model_cost_map(url=litellm.model_cost_map_url)
    )
    litellm.add_known_models(model_cost_map=litellm.model_cost)


def test_gpt5_supports_reasoning_effort(config: OpenAIConfig):
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt-5")
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt-5-mini")


def test_gpt5_chat_does_not_support_reasoning_effort(config: OpenAIConfig):
    assert "reasoning_effort" not in config.get_supported_openai_params(
        model="gpt-5-chat-latest"
    )


def test_gpt5_chat_supports_temperature(config: OpenAIConfig):
    params = config.map_openai_params(
        non_default_params={"temperature": 0.3},
        optional_params={},
        model="gpt-5-chat-latest",
        drop_params=False,
    )
    assert params["temperature"] == 0.3


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
    """Test that models supporting reasoning_effort='none' are correctly detected via model map."""
    # gpt-5.1 and gpt-5.2 chat variants support none
    assert gpt5_config._supports_reasoning_effort_level("gpt-5.1", "none")
    assert gpt5_config._supports_reasoning_effort_level("gpt-5.1-2025-11-13", "none")
    assert gpt5_config._supports_reasoning_effort_level("gpt-5.1-chat-latest", "none")
    assert gpt5_config._supports_reasoning_effort_level("gpt-5.2", "none")
    assert gpt5_config._supports_reasoning_effort_level("gpt-5.2-2025-12-11", "none")
    # codex/pro/chat variants do not support none
    assert not gpt5_config._supports_reasoning_effort_level("gpt-5.1-codex", "none")
    assert not gpt5_config._supports_reasoning_effort_level("gpt-5.1-codex-max", "none")
    assert not gpt5_config._supports_reasoning_effort_level(
        "gpt-5.2-chat-latest", "none"
    )
    assert not gpt5_config._supports_reasoning_effort_level("gpt-5.2-pro", "none")
    assert not gpt5_config._supports_reasoning_effort_level("gpt-5", "none")
    assert not gpt5_config._supports_reasoning_effort_level("gpt-5-mini", "none")
    assert not gpt5_config._supports_reasoning_effort_level("gpt-5-codex", "none")


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


def test_gpt5_4_allows_reasoning_effort_xhigh(config: OpenAIConfig):
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "xhigh"},
        optional_params={},
        model="gpt-5.4",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "xhigh"


def test_gpt5_4_pro_allows_reasoning_effort_xhigh(config: OpenAIConfig):
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "xhigh"},
        optional_params={},
        model="gpt-5.4-pro",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "xhigh"


def test_gpt5_4_mini_allows_reasoning_effort_xhigh(config: OpenAIConfig):
    """gpt-5.4-mini supports reasoning_effort='xhigh'."""
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "xhigh"},
        optional_params={},
        model="gpt-5.4-mini",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "xhigh"


def test_gpt5_4_nano_allows_reasoning_effort_xhigh(config: OpenAIConfig):
    """gpt-5.4-nano supports reasoning_effort='xhigh'."""
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "xhigh"},
        optional_params={},
        model="gpt-5.4-nano",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "xhigh"


def test_gpt5_4_nano_allows_reasoning_effort_none(config: OpenAIConfig):
    """gpt-5.4-nano supports reasoning_effort='none'."""
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "none"},
        optional_params={},
        model="gpt-5.4-nano",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "none"


def test_gpt5_4_mini_allows_reasoning_effort_none(config: OpenAIConfig):
    """gpt-5.4-mini supports reasoning_effort='none'."""
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "none"},
        optional_params={},
        model="gpt-5.4-mini",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "none"


def test_gpt5_4_allows_reasoning_effort_minimal(config: OpenAIConfig):
    """gpt-5.4 supports reasoning_effort='minimal'."""
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "minimal"},
        optional_params={},
        model="gpt-5.4",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "minimal"


def test_gpt5_4_pro_allows_reasoning_effort_minimal(config: OpenAIConfig):
    """gpt-5.4-pro supports reasoning_effort='minimal'."""
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "minimal"},
        optional_params={},
        model="gpt-5.4-pro",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "minimal"


def test_gpt5_4_mini_rejects_reasoning_effort_minimal(config: OpenAIConfig):
    """gpt-5.4-mini does not support reasoning_effort='minimal'."""
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"reasoning_effort": "minimal"},
            optional_params={},
            model="gpt-5.4-mini",
            drop_params=False,
        )


def test_gpt5_4_nano_rejects_reasoning_effort_minimal(config: OpenAIConfig):
    """gpt-5.4-nano does not support reasoning_effort='minimal'."""
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"reasoning_effort": "minimal"},
            optional_params={},
            model="gpt-5.4-nano",
            drop_params=False,
        )


def test_gpt5_4_mini_provider_prefixed_rejects_minimal(config: OpenAIConfig):
    """openai/gpt-5.4-mini correctly rejects minimal (model lookup normalizes prefix)."""
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"reasoning_effort": "minimal"},
            optional_params={},
            model="openai/gpt-5.4-mini",
            drop_params=False,
        )


def test_gpt5_drops_reasoning_effort_minimal_when_requested(config: OpenAIConfig):
    """reasoning_effort='minimal' is dropped for unsupported models when drop_params=True."""
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "minimal"},
        optional_params={},
        model="gpt-5.4-mini",
        drop_params=True,
    )
    assert "reasoning_effort" not in params


def test_gpt5_minimal_dict_triggers_validation(config: OpenAIConfig):
    """Dict with effort='minimal' triggers minimal model-support validation."""
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={
                "reasoning_effort": {"effort": "minimal", "summary": "detailed"}
            },
            optional_params={},
            model="gpt-5.4-mini",
            drop_params=False,
        )


def test_gpt5_minimal_dict_accepted_for_supported_model(config: OpenAIConfig):
    """Dict with effort='minimal' passes through for gpt-5.4+."""
    params = config.map_openai_params(
        non_default_params={
            "reasoning_effort": {"effort": "minimal", "summary": "detailed"}
        },
        optional_params={},
        model="gpt-5.4",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "minimal"


def test_gpt5_supports_reasoning_effort_level_minimal(gpt5_config: OpenAIGPT5Config):
    """Test that _supports_reasoning_effort_level correctly identifies minimal support."""
    assert gpt5_config._supports_reasoning_effort_level("gpt-5.4", "minimal")
    assert gpt5_config._supports_reasoning_effort_level("gpt-5.4-pro", "minimal")
    assert not gpt5_config._supports_reasoning_effort_level("gpt-5.4-mini", "minimal")
    assert not gpt5_config._supports_reasoning_effort_level("gpt-5.4-nano", "minimal")


def test_gpt5_minimal_explicitly_disabled_check(gpt5_config: OpenAIGPT5Config):
    """_is_reasoning_effort_level_explicitly_disabled returns True only for explicit False entries.

    Models with supports_minimal_reasoning_effort=false → disabled.
    Models with supports_minimal_reasoning_effort=true (or missing) → not disabled.
    Provider-prefixed models (openai/gpt-5.4-mini) are normalized before lookup.
    """
    assert gpt5_config._is_reasoning_effort_level_explicitly_disabled(
        "gpt-5.4-mini", "minimal"
    )
    assert gpt5_config._is_reasoning_effort_level_explicitly_disabled(
        "gpt-5.4-nano", "minimal"
    )
    assert gpt5_config._is_reasoning_effort_level_explicitly_disabled(
        "openai/gpt-5.4-mini", "minimal"
    )
    assert not gpt5_config._is_reasoning_effort_level_explicitly_disabled(
        "gpt-5.4", "minimal"
    )
    assert not gpt5_config._is_reasoning_effort_level_explicitly_disabled(
        "gpt-5.4-pro", "minimal"
    )


def test_is_explicitly_disabled_factory_minimal():
    """_is_explicitly_disabled_factory returns True only for explicit False entries.

    Verifies the shared helper used by _is_reasoning_effort_level_explicitly_disabled
    directly — so future changes to the helper are caught without going through the
    method wrapper.
    """
    key = "supports_minimal_reasoning_effort"
    assert _is_explicitly_disabled_factory("gpt-5.4-mini", None, key)
    assert _is_explicitly_disabled_factory("gpt-5.4-nano", None, key)
    assert _is_explicitly_disabled_factory("openai/gpt-5.4-mini", None, key)
    assert not _is_explicitly_disabled_factory("gpt-5.4", None, key)
    assert not _is_explicitly_disabled_factory("gpt-5.4-pro", None, key)
    assert not _is_explicitly_disabled_factory("gpt-5.4-turbo-preview", None, key)


def test_gpt5_unknown_model_passes_through_minimal(config: OpenAIConfig):
    """Unknown/unlisted gpt-5 models should pass reasoning_effort='minimal' through.

    Missing supports_minimal_reasoning_effort key is treated as supported,
    not as unsupported, to avoid breaking custom or newly-announced models.
    """
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "minimal"},
        optional_params={},
        model="gpt-5.4-turbo-preview",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "minimal"


def test_gpt5_5_pro_rejects_reasoning_effort_low(config: OpenAIConfig):
    """gpt-5.5-pro only accepts {medium, high, xhigh} — 'low' must raise.

    Verified against OpenAI's live API: /v1/chat/completions with
    reasoning_effort='low' on gpt-5.5-pro returns HTTP 400.
    """
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"reasoning_effort": "low"},
            optional_params={},
            model="gpt-5.5-pro",
            drop_params=False,
        )


def test_gpt5_5_pro_dated_rejects_reasoning_effort_low(config: OpenAIConfig):
    """Dated snapshot must inherit the base alias's low-rejection behavior."""
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"reasoning_effort": "low"},
            optional_params={},
            model="gpt-5.5-pro-2026-04-23",
            drop_params=False,
        )


def test_gpt5_5_pro_drops_reasoning_effort_low_when_requested(config: OpenAIConfig):
    """drop_params=True silently strips 'low' instead of round-tripping a 400."""
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "low"},
        optional_params={},
        model="gpt-5.5-pro",
        drop_params=True,
    )
    assert "reasoning_effort" not in params


def test_gpt5_5_chat_allows_reasoning_effort_low(config: OpenAIConfig):
    """gpt-5.5 (chat) supports 'low'; flag absent → opt-out check passes."""
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "low"},
        optional_params={},
        model="gpt-5.5",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "low"


def test_gpt5_unknown_model_passes_through_low(config: OpenAIConfig):
    """Unknown gpt-5 models pass 'low' through (opt-out, not opt-in)."""
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "low"},
        optional_params={},
        model="gpt-5.4-turbo-preview",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "low"


def test_gpt5_low_explicitly_disabled_check(gpt5_config: OpenAIGPT5Config):
    """supports_low_reasoning_effort=false → disabled; missing/true → not disabled."""
    assert gpt5_config._is_reasoning_effort_level_explicitly_disabled(
        "gpt-5.5-pro", "low"
    )
    assert gpt5_config._is_reasoning_effort_level_explicitly_disabled(
        "gpt-5.5-pro-2026-04-23", "low"
    )
    assert not gpt5_config._is_reasoning_effort_level_explicitly_disabled(
        "gpt-5.5", "low"
    )
    assert not gpt5_config._is_reasoning_effort_level_explicitly_disabled(
        "gpt-5.4", "low"
    )


def test_gpt5_normalizes_reasoning_effort_dict_with_summary(config: OpenAIConfig):
    """Dict with summary/generate_summary is normalized for chat completions."""
    params = config.map_openai_params(
        non_default_params={
            "reasoning_effort": {"effort": "high", "summary": "detailed"}
        },
        optional_params={},
        model="gpt-5.4",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "high"


def test_gpt5_xhigh_dict_triggers_validation(config: OpenAIConfig):
    """Dict with effort='xhigh' triggers xhigh model-support validation.

    Regression: when reasoning_effort is a dict, effective_effort must be used for
    the xhigh guard so validation is not silently skipped.
    """
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={
                "reasoning_effort": {"effort": "xhigh", "summary": "detailed"}
            },
            optional_params={},
            model="gpt-5.1",
            drop_params=False,
        )


def test_gpt5_xhigh_dict_accepted_for_supported_model(config: OpenAIConfig):
    """Dict with effort='xhigh' passes through for gpt-5.4+."""
    params = config.map_openai_params(
        non_default_params={
            "reasoning_effort": {"effort": "xhigh", "summary": "detailed"}
        },
        optional_params={},
        model="gpt-5.4",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "xhigh"


def test_gpt5_none_dict_with_tools_no_tool_drop(config: OpenAIConfig):
    """Dict with effort='none' and tools: no tool-drop, reasoning_effort preserved.

    Regression: effective_effort='none' must be used for tool-drop guard so
    {"effort": "none", "summary": "detailed"} is not incorrectly treated as non-none.
    """
    tools = [{"type": "function", "function": {"name": "test", "description": "test"}}]
    params = config.map_openai_params(
        non_default_params={
            "reasoning_effort": {"effort": "none", "summary": "detailed"},
            "tools": tools,
        },
        optional_params={},
        model="gpt-5.4",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "none"
    assert params["tools"] == tools


def test_gpt5_none_dict_with_sampling_params_allowed(config: OpenAIConfig):
    """Dict with effort='none' allows logprobs/top_p/top_logprobs.

    Regression: effective_effort='none' must be used for sampling guard so
    {"effort": "none", "summary": "detailed"} does not incorrectly trigger sampling errors.
    """
    params = config.map_openai_params(
        non_default_params={
            "reasoning_effort": {"effort": "none", "summary": "detailed"},
            "logprobs": True,
            "top_p": 0.9,
        },
        optional_params={},
        model="gpt-5.1",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "none"
    assert params["logprobs"] is True
    assert params["top_p"] == 0.9


def test_gpt5_normalizes_reasoning_effort_dict_with_summary_from_optional_params(
    config: OpenAIConfig,
):
    """reasoning_effort dict with summary in optional_params is normalized."""
    params = config.map_openai_params(
        non_default_params={},
        optional_params={
            "reasoning_effort": {"effort": "medium", "summary": "detailed"}
        },
        model="gpt-5.4",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "medium"


def test_gpt5_4_passes_through_reasoning_effort_with_tools(config: OpenAIConfig):
    """gpt-5.4 with tools + reasoning_effort: map_openai_params passes through both.

    Routing to Responses API (which supports tools + reasoning) happens at completion()
    level (responses_api_bridge_check). See test_responses_api_bridge_check_gpt_5_4_tools_plus_reasoning_routes_to_responses.
    """
    tools = [{"type": "function", "function": {"name": "test", "description": "test"}}]
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "high", "tools": tools},
        optional_params={},
        model="gpt-5.4",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "high"
    assert params["tools"] == tools


def test_gpt5_4_keeps_reasoning_effort_when_no_tools(config: OpenAIConfig):
    """reasoning_effort is kept when tools are not present."""
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "high"},
        optional_params={},
        model="gpt-5.4",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "high"


def test_gpt5_4_keeps_reasoning_effort_none_with_tools(config: OpenAIConfig):
    """reasoning_effort='none' is kept when tools are present."""
    tools = [{"type": "function", "function": {"name": "test", "description": "test"}}]
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "none", "tools": tools},
        optional_params={},
        model="gpt-5.4",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "none"
    assert params["tools"] == tools


def test_gpt5_2_keeps_reasoning_effort_with_tools(config: OpenAIConfig):
    """gpt-5.2: reasoning_effort drop only applies to gpt-5.4, not gpt-5.2."""
    tools = [{"type": "function", "function": {"name": "test", "description": "test"}}]
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "high", "tools": tools},
        optional_params={},
        model="gpt-5.2",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "high"
    assert params["tools"] == tools


def test_gpt5_4_pro_rejects_non_default_temperature(config: OpenAIConfig):
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"temperature": 0.5},
            optional_params={},
            model="gpt-5.4-pro",
            drop_params=False,
        )


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


def test_gpt5_1_temperature_with_reasoning_effort_in_optional_params(
    config: OpenAIConfig,
):
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


def test_gpt5_2_chat_temperature_restricted(config: OpenAIConfig):
    """Test that gpt-5.2-chat only supports temperature=1, like base gpt-5.

    Regression test for https://github.com/BerriAI/litellm/issues/21911
    """
    # gpt-5.2-chat should reject non-1 temperature when drop_params=False
    for model in ["gpt-5.2-chat", "gpt-5.2-chat-latest"]:
        with pytest.raises(litellm.utils.UnsupportedParamsError):
            config.map_openai_params(
                non_default_params={"temperature": 0.7},
                optional_params={},
                model=model,
                drop_params=False,
            )

        # temperature=1 should still work
        params = config.map_openai_params(
            non_default_params={"temperature": 1.0},
            optional_params={},
            model=model,
            drop_params=False,
        )
        assert params["temperature"] == 1.0

        # drop_params=True should silently drop non-1 temperature
        params = config.map_openai_params(
            non_default_params={"temperature": 0.5},
            optional_params={},
            model=model,
            drop_params=True,
        )
        assert "temperature" not in params
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


# GPT-5-Search specific tests
def test_gpt5_search_model_detection(gpt5_config: OpenAIGPT5Config):
    """Test that GPT-5 search models are correctly detected."""
    assert gpt5_config.is_model_gpt_5_search_model("gpt-5-search-api")
    assert gpt5_config.is_model_gpt_5_search_model("gpt-5-search-mini-api")

    assert not gpt5_config.is_model_gpt_5_search_model("gpt-5")
    assert not gpt5_config.is_model_gpt_5_search_model("gpt-5-codex")
    assert not gpt5_config.is_model_gpt_5_search_model("gpt-5-mini")


def test_gpt5_search_supported_params(gpt5_config: OpenAIGPT5Config):
    """Test that search models do NOT list reasoning/tool params as supported."""
    supported = gpt5_config.get_supported_openai_params(model="gpt-5-search-api")
    rejected = [
        "logit_bias",
        "modalities",
        "prediction",
        "n",
        "seed",
        "temperature",
        "tools",
        "tool_choice",
        "function_call",
        "functions",
        "parallel_tool_calls",
        "audio",
        "reasoning_effort",
    ]
    for param in rejected:
        assert (
            param not in supported
        ), f"{param} should not be supported for search models"


def test_gpt5_search_has_expected_params(gpt5_config: OpenAIGPT5Config):
    """Test that search models DO list the correct supported params."""
    supported = gpt5_config.get_supported_openai_params(model="gpt-5-search-api")
    expected = [
        "max_tokens",
        "max_completion_tokens",
        "stream",
        "stream_options",
        "web_search_options",
        "service_tier",
        "response_format",
        "user",
        "store",
        "verbosity",
        "extra_headers",
    ]
    for param in expected:
        assert param in supported, f"{param} should be supported for search models"


def test_gpt5_search_maps_max_tokens(config: OpenAIConfig):
    """Test that search models map max_tokens -> max_completion_tokens."""
    params = config.map_openai_params(
        non_default_params={"max_tokens": 200},
        optional_params={},
        model="gpt-5-search-api",
        drop_params=False,
    )
    assert params["max_completion_tokens"] == 200
    assert "max_tokens" not in params


def test_gpt5_search_drops_unsupported_params(config: OpenAIConfig):
    """Test that search models drop unsupported params via map_openai_params."""
    params = config.map_openai_params(
        non_default_params={
            "n": 2,
            "temperature": 0.7,
            "tools": [{"type": "function"}],
        },
        optional_params={},
        model="gpt-5-search-api",
        drop_params=True,
    )
    assert "n" not in params
    assert "temperature" not in params
    assert "tools" not in params


def test_gpt5_chat_strips_reasoning_summary_aliases_after_bridge_check(
    monkeypatch: pytest.MonkeyPatch,
):
    """Non-bridged GPT-5 chat calls strip Responses-only reasoning summary aliases."""
    captured_kwargs = {}

    def fake_openai_completion(**kwargs):
        captured_kwargs.update(kwargs)
        return {}

    monkeypatch.setattr(
        litellm_main.openai_chat_completions,
        "completion",
        fake_openai_completion,
    )

    litellm.completion(
        model="gpt-5",
        messages=[{"role": "user", "content": "ok"}],
        reasoning_effort="medium",
        reasoningSummary="auto",
        extra_body={"reasoning_summary": "ignored", "metadata": "ok"},
        api_key="fake-key",
    )

    optional_params = captured_kwargs["optional_params"]
    assert "reasoningSummary" not in optional_params
    assert "reasoning_summary" not in optional_params
    assert optional_params["extra_body"] == {"metadata": "ok"}


def test_reasoning_summary_alias_helpers_preserve_falsy_and_strip_all_aliases():
    optional_params = {"reasoningSummary": False, "reasoning_summary": "ignored"}

    assert peek_reasoning_summary_aliases(optional_params) is False
    stripped, rs_val = strip_reasoning_summary_aliases_from_optional_params(
        optional_params
    )

    assert rs_val is False
    assert stripped == {}

    optional_params = {
        "extra_body": {"reasoningSummary": False, "reasoning_summary": "ignored"}
    }

    assert peek_reasoning_summary_aliases(optional_params) is False
    stripped, rs_val = strip_reasoning_summary_aliases_from_optional_params(
        optional_params
    )

    assert rs_val is False
    assert stripped == {}

    optional_params = {
        "extra_body": {
            "reasoningSummary": "auto",
            "reasoning_summary": "ignored",
            "metadata": "ok",
        }
    }

    assert peek_reasoning_summary_aliases(optional_params) == "auto"
    stripped, rs_val = strip_reasoning_summary_aliases_from_optional_params(
        optional_params
    )

    assert rs_val == "auto"
    assert stripped == {"extra_body": {"metadata": "ok"}}


# GPT-5 unsupported params audit (validated via direct API calls)
def test_gpt5_rejects_params_unsupported_by_openai(config: OpenAIConfig):
    """Params that OpenAI rejects for all GPT-5 reasoning models."""
    rejected_params = [
        "logit_bias",
        "modalities",
        "prediction",
        "audio",
        "web_search_options",
    ]
    for model in ["gpt-5", "gpt-5-mini", "gpt-5-codex", "gpt-5.1", "gpt-5.2"]:
        supported = config.get_supported_openai_params(model=model)
        for param in rejected_params:
            assert (
                param not in supported
            ), f"{param} should not be supported for {model}"


def test_gpt5_1_supports_logprobs_top_p(config: OpenAIConfig):
    """gpt-5.1/5.2 support logprobs, top_p, top_logprobs when reasoning_effort='none'."""
    for model in ["gpt-5.1", "gpt-5.2"]:
        supported = config.get_supported_openai_params(model=model)
        assert "logprobs" in supported, f"logprobs should be supported for {model}"
        assert "top_p" in supported, f"top_p should be supported for {model}"
        assert (
            "top_logprobs" in supported
        ), f"top_logprobs should be supported for {model}"


def test_gpt5_base_does_not_support_logprobs_top_p(config: OpenAIConfig):
    """Base gpt-5/gpt-5-mini do NOT support logprobs, top_p, top_logprobs."""
    for model in ["gpt-5", "gpt-5-mini", "gpt-5-codex"]:
        supported = config.get_supported_openai_params(model=model)
        assert (
            "logprobs" not in supported
        ), f"logprobs should not be supported for {model}"
        assert "top_p" not in supported, f"top_p should not be supported for {model}"
        assert (
            "top_logprobs" not in supported
        ), f"top_logprobs should not be supported for {model}"


def test_gpt5_1_logprobs_passthrough(config: OpenAIConfig):
    """Test that logprobs passes through for gpt-5.1."""
    params = config.map_openai_params(
        non_default_params={"logprobs": True, "top_logprobs": 3},
        optional_params={},
        model="gpt-5.1",
        drop_params=False,
    )
    assert params["logprobs"] is True
    assert params["top_logprobs"] == 3


def test_gpt5_1_top_p_passthrough(config: OpenAIConfig):
    """Test that top_p passes through for gpt-5.1."""
    params = config.map_openai_params(
        non_default_params={"top_p": 0.9},
        optional_params={},
        model="gpt-5.1",
        drop_params=False,
    )
    assert params["top_p"] == 0.9


def test_gpt5_1_logprobs_rejected_with_reasoning_effort(config: OpenAIConfig):
    """logprobs/top_p/top_logprobs are rejected when reasoning_effort != 'none'."""
    for effort in ["low", "medium", "high"]:
        with pytest.raises(litellm.utils.UnsupportedParamsError):
            config.map_openai_params(
                non_default_params={"logprobs": True, "reasoning_effort": effort},
                optional_params={},
                model="gpt-5.1",
                drop_params=False,
            )


def test_gpt5_1_top_p_rejected_with_reasoning_effort(config: OpenAIConfig):
    """top_p is rejected when reasoning_effort != 'none'."""
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"top_p": 0.9, "reasoning_effort": "high"},
            optional_params={},
            model="gpt-5.1",
            drop_params=False,
        )


def test_gpt5_1_logprobs_dropped_with_reasoning_effort(config: OpenAIConfig):
    """logprobs/top_p are dropped when reasoning_effort != 'none' and drop_params=True."""
    params = config.map_openai_params(
        non_default_params={"logprobs": True, "top_p": 0.9, "reasoning_effort": "high"},
        optional_params={},
        model="gpt-5.1",
        drop_params=True,
    )
    assert "logprobs" not in params
    assert "top_p" not in params
    assert params["reasoning_effort"] == "high"


# ---------------------------------------------------------------------------
# Responses API: GPT-5 temperature validation (#16090)
# ---------------------------------------------------------------------------

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams


@pytest.fixture()
def responses_config() -> OpenAIResponsesAPIConfig:
    return OpenAIResponsesAPIConfig()


def test_responses_gpt5_drop_temperature(
    responses_config: OpenAIResponsesAPIConfig,
):
    """drop_params=True should silently drop temperature!=1 for gpt-5."""
    params = responses_config.map_openai_params(
        response_api_optional_params=ResponsesAPIOptionalRequestParams(
            temperature=0.5,
        ),
        model="gpt-5",
        drop_params=True,
    )
    assert "temperature" not in params


def test_responses_gpt5_reject_temperature(
    responses_config: OpenAIResponsesAPIConfig,
):
    """Without drop_params, temperature!=1 should raise UnsupportedParamsError."""
    with pytest.raises(litellm.UnsupportedParamsError):
        responses_config.map_openai_params(
            response_api_optional_params=ResponsesAPIOptionalRequestParams(
                temperature=0.5,
            ),
            model="gpt-5",
            drop_params=False,
        )


def test_responses_gpt5_allow_temperature_1(
    responses_config: OpenAIResponsesAPIConfig,
):
    """temperature=1 should always be allowed for gpt-5."""
    params = responses_config.map_openai_params(
        response_api_optional_params=ResponsesAPIOptionalRequestParams(
            temperature=1,
        ),
        model="gpt-5",
        drop_params=False,
    )
    assert params["temperature"] == 1


def test_responses_gpt5_mini_drop_temperature(
    responses_config: OpenAIResponsesAPIConfig,
):
    """gpt-5-mini should also drop temperature!=1."""
    params = responses_config.map_openai_params(
        response_api_optional_params=ResponsesAPIOptionalRequestParams(
            temperature=0.7,
        ),
        model="gpt-5-mini",
        drop_params=True,
    )
    assert "temperature" not in params


def test_responses_gpt5_chat_allow_temperature(
    responses_config: OpenAIResponsesAPIConfig,
):
    """gpt-5-chat models should allow any temperature (not GPT-5 restricted)."""
    params = responses_config.map_openai_params(
        response_api_optional_params=ResponsesAPIOptionalRequestParams(
            temperature=0.3,
        ),
        model="gpt-5-chat-latest",
        drop_params=False,
    )
    assert params["temperature"] == 0.3


def test_responses_gpt51_allow_temperature_no_reasoning(
    responses_config: OpenAIResponsesAPIConfig,
):
    """gpt-5.1 supports reasoning_effort='none'; no reasoning defaults to 'none',
    so temperature should be allowed."""
    params = responses_config.map_openai_params(
        response_api_optional_params=ResponsesAPIOptionalRequestParams(
            temperature=0.5,
        ),
        model="gpt-5.1",
        drop_params=False,
    )
    assert params["temperature"] == 0.5


def test_responses_gpt51_drop_temperature_with_high_effort(
    responses_config: OpenAIResponsesAPIConfig,
):
    """gpt-5.1 with reasoning.effort='high' should drop temperature!=1."""
    params = responses_config.map_openai_params(
        response_api_optional_params=ResponsesAPIOptionalRequestParams(
            temperature=0.5,
            reasoning={"effort": "high"},
        ),
        model="gpt-5.1",
        drop_params=True,
    )
    assert "temperature" not in params


def test_responses_gpt54_allow_temperature_effort_none(
    responses_config: OpenAIResponsesAPIConfig,
):
    """gpt-5.4 with explicit reasoning.effort='none' should allow temperature."""
    params = responses_config.map_openai_params(
        response_api_optional_params=ResponsesAPIOptionalRequestParams(
            temperature=0.7,
            reasoning={"effort": "none"},
        ),
        model="gpt-5.4",
        drop_params=False,
    )
    assert params["temperature"] == 0.7
