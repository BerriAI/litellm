"""Tests for ``reasoning_effort`` translation on the Anthropic /v1/messages route."""

import pytest

import litellm
from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
)
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeMessagesConfig,
)


@pytest.fixture
def local_model_cost_map(monkeypatch):
    """Force the bundled backup cost map so Opus 4.8 adaptive detection (driven
    by the ``supports_adaptive_thinking`` flag) doesn't depend on the
    network-fetched ``main`` copy, which lacks the flag until this branch merges."""
    original = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original
        litellm.get_model_info.cache_clear()


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
    "model,effort,expected_effort",
    [
        ("invoke/us.anthropic.claude-opus-4-6-v1", "xhigh", "max"),
        ("invoke/us.anthropic.claude-opus-4-6-v1", "max", "max"),
        ("invoke/us.anthropic.claude-opus-4-6-v1", "high", "high"),
        ("invoke/us.anthropic.claude-opus-4-7", "xhigh", "xhigh"),
    ],
)
def test_bedrock_invoke_messages_clamps_effort_to_ceiling(
    local_model_cost_map, model, effort, expected_effort
):
    """Bedrock Invoke /v1/messages degrades effort to the model's ceiling.

    Claude Code "goal mode" sends ``xhigh``; Opus 4.6 must clamp to ``max``
    instead of raising, while Opus 4.7 (ceiling ``xhigh``) keeps ``xhigh``.
    """
    config = AmazonAnthropicClaudeMessagesConfig()
    optional_params = {"max_tokens": 1024, "reasoning_effort": effort}

    result = config.transform_anthropic_messages_request(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert result["output_config"]["effort"] == expected_effort
    assert result["thinking"]["type"] == "adaptive"


def test_bedrock_invoke_messages_rejects_xhigh_without_ceiling(local_model_cost_map):
    """Sonnet 4.6 on Bedrock has no effort ceiling, so xhigh is still rejected."""
    config = AmazonAnthropicClaudeMessagesConfig()
    optional_params = {"max_tokens": 1024, "reasoning_effort": "xhigh"}

    with pytest.raises(AnthropicError) as exc_info:
        config.transform_anthropic_messages_request(
            model="invoke/us.anthropic.claude-sonnet-4-6",
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
def test_reasoning_effort_max_accepted_on_sonnet_46_messages(
    local_model_cost_map, model
):
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


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-4-6",
        "bedrock/invoke/us.anthropic.claude-sonnet-4-6",
        "vertex_ai/claude-sonnet-4-6",
        "claude-opus-4-6",
        "bedrock/invoke/us.anthropic.claude-opus-4-6-v1:0",
        "vertex_ai/claude-opus-4-6",
    ],
)
def test_legacy_thinking_high_budget_clamps_to_high_when_xhigh_unsupported(
    local_model_cost_map, model
):
    """Claude Code sends ``thinking.budget_tokens=31999``; Sonnet 4.6 and Opus 4.6
    have no ``xhigh`` tier, so the translator must emit ``high`` rather than the
    provider-invalid ``xhigh`` (regression for issue #29282)."""
    config = AnthropicMessagesConfig()
    optional_params = {
        "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 31999},
    }

    result = config.transform_anthropic_messages_request(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert result.get("thinking") == {"type": "adaptive"}
    assert result.get("output_config") == {"effort": "high"}


def test_legacy_thinking_high_budget_keeps_xhigh_when_supported():
    """Opus 4.7 advertises an ``xhigh`` tier, so the high-budget bucket keeps it."""
    config = AnthropicMessagesConfig()
    optional_params = {
        "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 31999},
    }

    result = config.transform_anthropic_messages_request(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert result.get("thinking") == {"type": "adaptive"}
    assert result.get("output_config") == {"effort": "xhigh"}


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-8",
        "bedrock/us.anthropic.claude-opus-4-8",
        "bedrock/invoke/us.anthropic.claude-opus-4-8",
    ],
)
def test_legacy_thinking_translates_to_adaptive_for_opus_48(
    model, local_model_cost_map
):
    """Regression for issue #29188: Opus 4.8 requires adaptive thinking, but the
    legacy ``thinking.type='enabled'`` shape was passed through unchanged for
    Bedrock 4.8 (its cost-map entry lacked ``supports_adaptive_thinking`` and the
    lookup didn't strip the provider prefix), so Bedrock rejected the request. The
    reporter's reproducer used ``budget_tokens=24000``, the ``xhigh`` bucket."""
    config = AnthropicMessagesConfig()
    optional_params = {
        "max_tokens": 100,
        "thinking": {"type": "enabled", "budget_tokens": 24000},
    }

    result = config.transform_anthropic_messages_request(
        model=model,
        messages=[{"role": "user", "content": "ping"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert result.get("thinking") == {"type": "adaptive"}
    assert result.get("output_config") == {"effort": "xhigh"}


@pytest.mark.parametrize(
    "budget_tokens,expected_effort",
    [
        (DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET * 2, "high"),
        (DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET, "high"),
        (DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET, "high"),
        (DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET - 1, "medium"),
        (DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET, "medium"),
        (DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET - 1, "low"),
        (1, "low"),
    ],
)
def test_legacy_thinking_budget_buckets_on_sonnet_46(budget_tokens, expected_effort):
    config = AnthropicMessagesConfig()
    optional_params = {
        "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": budget_tokens},
    }

    result = config.transform_anthropic_messages_request(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert result.get("output_config") == {"effort": expected_effort}


def test_legacy_thinking_does_not_override_explicit_output_config():
    config = AnthropicMessagesConfig()
    optional_params = {
        "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 31999},
        "output_config": {"effort": "low"},
    }

    result = config.transform_anthropic_messages_request(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert result.get("output_config") == {"effort": "low"}


def test_legacy_thinking_left_untouched_on_non_adaptive_model():
    config = AnthropicMessagesConfig()
    optional_params = {
        "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 31999},
    }

    result = config.transform_anthropic_messages_request(
        model="claude-opus-4-5",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert result.get("thinking") == {"type": "enabled", "budget_tokens": 31999}
    assert "output_config" not in result
