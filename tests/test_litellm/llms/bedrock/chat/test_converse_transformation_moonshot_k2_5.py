"""
Unit tests for Bedrock Moonshot K2.5 reasoning translation on Converse API.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig


def test_is_moonshot_kimi_k2_5_model():
    config = AmazonConverseConfig()

    assert config._is_moonshot_kimi_k2_5_model("moonshotai.kimi-k2.5") is True
    assert (
        config._is_moonshot_kimi_k2_5_model("bedrock/moonshotai.kimi-k2.5") is True
    )
    assert (
        config._is_moonshot_kimi_k2_5_model("bedrock/us-east-1/moonshotai.kimi-k2.5")
        is True
    )
    assert (
        config._is_moonshot_kimi_k2_5_model(
            "bedrock/converse/us-east-1/moonshotai.kimi-k2.5"
        )
        is True
    )

    assert config._is_moonshot_kimi_k2_5_model("moonshotai.kimi-k2-thinking") is False
    assert (
        config._is_moonshot_kimi_k2_5_model("anthropic.claude-3-5-sonnet-20241022-v2:0")
        is False
    )


def test_map_openai_params_moonshot_thinking_to_reasoning_config():
    config = AmazonConverseConfig()

    result = config.map_openai_params(
        non_default_params={"thinking": {"type": "enabled", "budget_tokens": 2048}},
        optional_params={},
        model="bedrock/us-east-1/moonshotai.kimi-k2.5",
        drop_params=False,
    )

    assert result["reasoning_config"] == "high"
    assert "thinking" not in result


@pytest.mark.parametrize(
    "reasoning_effort,expected",
    [
        ("minimal", "high"),
        ("low", "high"),
        ("medium", "high"),
        ("high", "high"),
        ("xhigh", "high"),
        ("custom-value", "high"),
        (True, "high"),
        ({"effort": "low", "summary": "auto"}, "high"),
        ({"effort": "none", "summary": "auto"}, None),
        ({"type": "enabled"}, "high"),
        ({"type": "disabled"}, None),
        ({"enabled": False}, None),
        ({"enabled": True}, "high"),
        ({}, "high"),
    ],
)
def test_map_openai_params_moonshot_reasoning_effort_to_reasoning_config(
    reasoning_effort, expected
):
    config = AmazonConverseConfig()

    result = config.map_openai_params(
        non_default_params={"reasoning_effort": reasoning_effort},
        optional_params={},
        model="bedrock/us-east-1/moonshotai.kimi-k2.5",
        drop_params=False,
    )

    if expected is None:
        assert "reasoning_config" not in result
    else:
        assert result["reasoning_config"] == expected
    assert "reasoning_effort" not in result
    assert "thinking" not in result


@pytest.mark.parametrize("reasoning_effort", ["none", "false", "disabled", False, None])
def test_map_openai_params_moonshot_reasoning_effort_disable_values_disable_reasoning(
    reasoning_effort,
):
    config = AmazonConverseConfig()

    result = config.map_openai_params(
        non_default_params={"reasoning_effort": reasoning_effort},
        optional_params={},
        model="bedrock/us-east-1/moonshotai.kimi-k2.5",
        drop_params=False,
    )

    if reasoning_effort is None:
        assert result["reasoning_config"] == "high"
    else:
        assert "reasoning_config" not in result
    assert "thinking" not in result
    assert "reasoning_effort" not in result


def test_prepare_request_params_translates_legacy_thinking_for_moonshot():
    config = AmazonConverseConfig()

    _, additional_request_params, _ = config._prepare_request_params(
        optional_params={"thinking": {"type": "enabled"}},
        model="bedrock/us-east-1/moonshotai.kimi-k2.5",
    )

    assert additional_request_params["reasoning_config"] == "high"
    assert "thinking" not in additional_request_params
    assert "reasoning_effort" not in additional_request_params


def test_prepare_request_params_defaults_reasoning_enabled_for_moonshot():
    config = AmazonConverseConfig()

    _, additional_request_params, _ = config._prepare_request_params(
        optional_params={},
        model="bedrock/us-east-1/moonshotai.kimi-k2.5",
    )

    assert additional_request_params["reasoning_config"] == "high"


def test_prepare_request_params_translates_openai_reasoning_effort_dict_for_moonshot():
    config = AmazonConverseConfig()

    _, additional_request_params, _ = config._prepare_request_params(
        optional_params={"reasoning_effort": {"effort": "high", "summary": "auto"}},
        model="bedrock/us-east-1/moonshotai.kimi-k2.5",
    )

    assert additional_request_params["reasoning_config"] == "high"
    assert "thinking" not in additional_request_params
    assert "reasoning_effort" not in additional_request_params


def test_prepare_request_params_thinking_disabled_does_not_set_reasoning_config():
    config = AmazonConverseConfig()

    _, additional_request_params, _ = config._prepare_request_params(
        optional_params={"thinking": {"type": "disabled"}},
        model="bedrock/us-east-1/moonshotai.kimi-k2.5",
    )

    assert "reasoning_config" not in additional_request_params
    assert "thinking" not in additional_request_params


@pytest.mark.parametrize(
    "disable_value",
    [
        "none",
        "false",
        "disabled",
        False,
    ],
)
def test_prepare_request_params_reasoning_effort_disable_values_keep_reasoning_disabled(
    disable_value,
):
    config = AmazonConverseConfig()

    _, additional_request_params, _ = config._prepare_request_params(
        optional_params={"reasoning_effort": disable_value},
        model="bedrock/us-east-1/moonshotai.kimi-k2.5",
    )

    assert "reasoning_config" not in additional_request_params
    assert "reasoning_effort" not in additional_request_params
