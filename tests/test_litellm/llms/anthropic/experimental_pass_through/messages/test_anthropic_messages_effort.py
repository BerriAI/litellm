from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
)
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)


def _claude_code_payload(effort="medium", max_tokens=8192, **output_config_extra):
    """The exact adaptive-thinking shape Claude Code (claude-cli) sends."""
    output_config = {"effort": effort, **output_config_extra}
    return {
        "max_tokens": max_tokens,
        "thinking": {"type": "adaptive"},
        "output_config": output_config,
    }


def _transform(model, params, litellm_params=None):
    return AnthropicMessagesConfig().transform_anthropic_messages_request(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=dict(params),
        litellm_params=litellm_params or {},
        headers={},
    )


def test_effort_translated_to_legacy_thinking_for_haiku_4_5():
    """Core regression: Claude Code sends adaptive thinking + effort to Haiku 4.5
    (thinking-capable, pre-4.6). Effort must be translated to legacy extended
    thinking rather than forwarded raw (which Anthropic rejects with "This model
    does not support the effort parameter")."""
    result = _transform("claude-haiku-4-5", _claude_code_payload(effort="medium"))

    assert result["thinking"] == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    }
    assert "output_config" not in result


def test_effort_high_maps_to_high_budget_for_sonnet_4_5():
    result = _transform("claude-sonnet-4-5", _claude_code_payload(effort="high"))

    assert result["thinking"] == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    }
    assert "output_config" not in result


def test_adaptive_effort_passes_through_untouched_for_4_6():
    """4.6+ natively supports the adaptive interface, so it must not be rewritten."""
    result = _transform("claude-sonnet-4-6", _claude_code_payload(effort="high"))

    assert result["thinking"] == {"type": "adaptive"}
    assert result["output_config"] == {"effort": "high"}


def test_thinking_and_effort_dropped_for_non_reasoning_model():
    """A model with no reasoning support cannot take thinking or effort, so both are
    silently dropped (no drop_params required) so the request still succeeds."""
    result = _transform("claude-3-5-haiku-latest", _claude_code_payload(effort="medium"))

    assert "thinking" not in result
    assert "output_config" not in result


def test_residual_output_config_preserved_after_effort_translation():
    """output_config may carry `format` (structured outputs) alongside effort. Only
    the consumed effort key is removed; the residual is left for provider subclasses
    (bedrock/vertex) to handle, and effort is translated to legacy thinking."""
    result = _transform(
        "claude-haiku-4-5",
        _claude_code_payload(effort="medium", format={"type": "json_schema"}),
    )

    assert result["thinking"] == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    }
    assert result["output_config"] == {"format": {"type": "json_schema"}}


def test_budget_capped_below_max_tokens():
    """Adaptive thinking carries no budget, so the translated legacy budget must be
    capped below max_tokens (Anthropic requires max_tokens > budget_tokens). A
    high-effort budget (4096) with max_tokens=3000 must be capped to 2999."""
    result = _transform("claude-haiku-4-5", _claude_code_payload(effort="high", max_tokens=3000))

    assert result["thinking"] == {"type": "enabled", "budget_tokens": 2999}


def test_thinking_dropped_when_max_tokens_too_small_for_min_budget():
    """When max_tokens can't fit even the minimum thinking budget, thinking is
    silently dropped so the request still succeeds rather than being rejected."""
    result = _transform("claude-haiku-4-5", _claude_code_payload(effort="medium", max_tokens=512))

    assert "thinking" not in result
    assert "output_config" not in result


def test_non_adaptive_request_without_effort_is_untouched():
    """A non-adaptive model receiving a request with no adaptive interface (no
    effort, no adaptive thinking) must pass through untouched."""
    result = AnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 1024},
        litellm_params={},
        headers={},
    )

    assert "thinking" not in result
    assert "output_config" not in result
