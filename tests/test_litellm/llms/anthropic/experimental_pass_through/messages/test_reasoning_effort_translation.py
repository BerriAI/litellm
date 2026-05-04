"""Tests for ``reasoning_effort`` translation on the Anthropic /v1/messages route."""

import pytest

from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)


@pytest.mark.parametrize(
    "reasoning_effort,expected_effort",
    [
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
        ("max", "max"),
    ],
)
def test_reasoning_effort_maps_to_output_config_for_adaptive_model(
    reasoning_effort, expected_effort
):
    config = AnthropicMessagesConfig()
    optional_params = {"max_tokens": 1024, "reasoning_effort": reasoning_effort}

    result = config.transform_anthropic_messages_request(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert "reasoning_effort" not in result
    assert result.get("thinking") == {"type": "adaptive"}
    assert result.get("output_config") == {"effort": expected_effort}


def test_reasoning_effort_none_clears_thinking_and_output_config():
    config = AnthropicMessagesConfig()
    optional_params = {
        "max_tokens": 1024,
        "reasoning_effort": "none",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }

    result = config.transform_anthropic_messages_request(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert "reasoning_effort" not in result
    assert "thinking" not in result
    assert "output_config" not in result


def test_reasoning_effort_on_non_adaptive_model_uses_thinking_budget():
    config = AnthropicMessagesConfig()
    optional_params = {"max_tokens": 1024, "reasoning_effort": "high"}

    result = config.transform_anthropic_messages_request(
        model="claude-opus-4-5",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert "reasoning_effort" not in result
    assert "output_config" not in result
    thinking = result.get("thinking")
    assert isinstance(thinking, dict)
    assert thinking.get("type") == "enabled"
    assert isinstance(thinking.get("budget_tokens"), int)
    assert thinking["budget_tokens"] >= 1024


@pytest.mark.parametrize("bad_effort", ["invalid", "disabled", ""])
def test_invalid_reasoning_effort_raises_400(bad_effort):
    config = AnthropicMessagesConfig()
    optional_params = {"max_tokens": 1024, "reasoning_effort": bad_effort}

    with pytest.raises(AnthropicError) as exc_info:
        config.transform_anthropic_messages_request(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "Hello"}],
            anthropic_messages_optional_request_params=optional_params,
            litellm_params={},
            headers={},
        )

    assert exc_info.value.status_code == 400


@pytest.mark.parametrize(
    "model,bad_effort",
    [
        ("claude-opus-4-6", "xhigh"),
        ("bedrock/invoke/us.anthropic.claude-opus-4-6-v1", "xhigh"),
        ("claude-sonnet-4-6", "xhigh"),
    ],
)
def test_reasoning_effort_unsupported_tier_raises_400_messages(model, bad_effort):
    config = AnthropicMessagesConfig()
    optional_params = {"max_tokens": 1024, "reasoning_effort": bad_effort}

    with pytest.raises(AnthropicError) as exc_info:
        config.transform_anthropic_messages_request(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            anthropic_messages_optional_request_params=optional_params,
            litellm_params={},
            headers={},
        )

    assert exc_info.value.status_code == 400
    assert "not supported by this model" in str(exc_info.value)


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-4-6",
        "bedrock/invoke/us.anthropic.claude-sonnet-4-6",
    ],
)
def test_reasoning_effort_max_accepted_on_sonnet_46_messages(model):
    config = AnthropicMessagesConfig()
    optional_params = {"max_tokens": 1024, "reasoning_effort": "max"}

    result = config.transform_anthropic_messages_request(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={},
    )

    output_config = result.get("output_config")
    assert isinstance(output_config, dict) and output_config.get("effort") == "max"


def test_explicit_output_config_wins_over_reasoning_effort():
    config = AnthropicMessagesConfig()
    optional_params = {
        "max_tokens": 1024,
        "reasoning_effort": "low",
        "output_config": {"effort": "max"},
    }

    result = config.transform_anthropic_messages_request(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert "reasoning_effort" not in result
    assert result.get("output_config") == {"effort": "max"}


def test_explicit_thinking_wins_over_reasoning_effort():
    config = AnthropicMessagesConfig()
    optional_params = {
        "max_tokens": 1024,
        "reasoning_effort": "low",
        "thinking": {"type": "enabled", "budget_tokens": 8000},
    }

    result = config.transform_anthropic_messages_request(
        model="claude-opus-4-5",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert "reasoning_effort" not in result
    assert result.get("thinking") == {"type": "enabled", "budget_tokens": 8000}


def test_reasoning_effort_in_supported_params():
    config = AnthropicMessagesConfig()
    assert "reasoning_effort" in config.get_supported_anthropic_messages_params(
        "claude-opus-4-7"
    )
