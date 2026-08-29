"""Tests for ``reasoning_effort`` translation on the Anthropic /v1/messages route."""

import pytest

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
    assert result.get("thinking") == {"type": "adaptive", "display": "summarized"}
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
        "claude-opus-4-6",
        "claude-sonnet-4-6-20260219",
        "bedrock/invoke/us.anthropic.claude-sonnet-4-6",
        "bedrock/invoke/us.anthropic.claude-opus-4-6-v1:0",
        "vertex_ai/claude-sonnet-4-6",
        "vertex_ai/claude-opus-4-6",
        "azure_ai/claude-sonnet-4-6",
    ],
)
def test_legacy_thinking_budget_preserved_verbatim_on_46(local_model_cost_map, model):
    """Regression for the passthrough silently dropping a caller's hard thinking
    budget: the 4.6 family accepts ``thinking.type=enabled`` with ``budget_tokens``
    natively, so rewriting it to ``thinking.type=adaptive`` + ``output_config.effort``
    (which carries no ceiling) let reasoning run past the requested cap. The legacy
    shape must be forwarded verbatim, in every 4.6 id shape including unmapped dated
    releases resolved by the ``claude-legacy-thinking`` fallback rule."""
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

    assert result.get("thinking") == {"type": "enabled", "budget_tokens": 31999}
    assert "output_config" not in result


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
    "model,expected_effort",
    [
        ("claude-sonnet-5", "xhigh"),
        ("claude-opus-5", "xhigh"),
        ("claude-newfamily-6", "high"),
    ],
)
def test_legacy_thinking_translates_to_adaptive_for_5_and_future_models(
    local_model_cost_map, model, expected_effort
):
    """The 5 families reject ``thinking.type=enabled``, so the adaptive translation
    stays the safe default for every adaptive model not flagged
    ``supports_legacy_thinking``, unmapped future ids included. An unmapped id
    cannot prove ``xhigh`` support, so its high-budget bucket clamps to ``high``."""
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
    assert result.get("output_config") == {"effort": expected_effort}


@pytest.mark.parametrize(
    "budget_tokens,expected_effort",
    [
        (DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET * 2, "xhigh"),
        (DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET, "xhigh"),
        (DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET, "high"),
        (DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET - 1, "medium"),
        (DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET, "medium"),
        (DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET - 1, "low"),
        (1, "low"),
    ],
)
def test_legacy_thinking_budget_buckets_on_opus_48(
    local_model_cost_map, budget_tokens, expected_effort
):
    config = AnthropicMessagesConfig()
    optional_params = {
        "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": budget_tokens},
    }

    result = config.transform_anthropic_messages_request(
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert result.get("output_config") == {"effort": expected_effort}


def test_legacy_thinking_does_not_override_explicit_output_config(local_model_cost_map):
    config = AnthropicMessagesConfig()
    optional_params = {
        "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 31999},
        "output_config": {"effort": "low"},
    }

    result = config.transform_anthropic_messages_request(
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert result.get("thinking") == {"type": "adaptive"}
    assert result.get("output_config") == {"effort": "low"}


def test_legacy_thinking_with_explicit_output_config_untouched_on_46(
    local_model_cost_map,
):
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

    assert result.get("thinking") == {"type": "enabled", "budget_tokens": 31999}
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


@pytest.mark.parametrize(
    "model, expected_dropped",
    [
        ("claude-fable-5", True),
        ("claude-opus-5", False),
        ("claude-sonnet-4-5", False),
    ],
)
def test_disabled_thinking_omitted_for_always_on_models_messages(
    local_model_cost_map, model, expected_dropped
):
    """/v1/messages: ``thinking={"type": "disabled"}`` is omitted for always-on-thinking
    models and forwarded verbatim for models that accept it."""
    config = AnthropicMessagesConfig()
    optional_params = {"max_tokens": 64, "thinking": {"type": "disabled"}}

    result = config.transform_anthropic_messages_request(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={},
    )

    if expected_dropped:
        assert "thinking" not in result
    else:
        assert result["thinking"] == {"type": "disabled"}
