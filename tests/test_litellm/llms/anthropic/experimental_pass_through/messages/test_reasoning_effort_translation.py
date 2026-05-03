"""
Tests for OpenAI-style ``reasoning_effort`` translation on the Anthropic
/v1/messages route.

The /v1/messages spec doesn't include ``reasoning_effort`` — without
translation it gets silently dropped at the filter step, leaving every
adaptive tier collapsed to the same behavior on Bedrock Invoke /v1/messages
(and on Anthropic / Azure AI / Vertex AI when callers pass it on the
messages route).

These tests pin the translation and validation behavior at the shared
``AnthropicMessagesConfig`` level so all four /v1/messages routes
(direct Anthropic, Azure AI, Vertex AI, Bedrock Invoke) inherit the
same mapping.
"""

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
    """
    For Claude 4.6 / 4.7, ``reasoning_effort`` is mapped to
    ``thinking={"type": "adaptive"}`` plus ``output_config.effort=<tier>``,
    using the same mapping table as the chat completion path so the two
    routes can't drift.
    """
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
    """``reasoning_effort='none'`` opts out of extended thinking entirely."""
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
    """
    Non-adaptive models (Opus 4.5 / earlier) take ``thinking.budget_tokens``
    rather than ``output_config.effort``. The translation falls back to
    the budget mapping in that case.
    """
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
    """
    Garbage ``reasoning_effort`` values surface as a clean 400 instead of
    silently passing through to the provider as an unknown
    ``output_config.effort`` (which would 500).
    """
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


def test_explicit_output_config_wins_over_reasoning_effort():
    """
    Explicit native ``output_config.effort`` is never overridden by the
    OpenAI alias. Same precedence as
    ``_translate_legacy_thinking_for_adaptive_model``.
    """
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
    """Explicit native ``thinking`` is never overridden by the alias."""
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
    """``reasoning_effort`` is advertised as a supported messages param so
    callers and validation paths can introspect the schema."""
    config = AnthropicMessagesConfig()
    assert "reasoning_effort" in config.get_supported_anthropic_messages_params(
        "claude-opus-4-7"
    )
