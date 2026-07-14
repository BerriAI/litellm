import pytest

from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
)
from litellm.llms.anthropic.common_utils import AnthropicError
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


def test_opus_4_5_keeps_effort_but_drops_adaptive_thinking():
    """Regression: Opus 4.5 advertises supports_output_config (accepts
    output_config.effort) but is NOT adaptive, so thinking:{type:adaptive} is
    rejected by Anthropic. The effort must be kept and only the adaptive thinking
    block dropped, rather than early-returning and forwarding adaptive thinking raw."""
    result = _transform("claude-opus-4-5", _claude_code_payload(effort="medium"))

    assert result["output_config"] == {"effort": "medium"}
    assert "thinking" not in result


def test_opus_4_5_preserves_native_effort_without_adaptive_thinking():
    """A caller sending output_config.effort alone (no adaptive thinking) to Opus 4.5
    must pass through untouched, since the model supports it natively."""
    result = AnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-opus-4-5",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 8192,
            "output_config": {"effort": "high"},
        },
        litellm_params={},
        headers={},
    )

    assert result["output_config"] == {"effort": "high"}
    assert "thinking" not in result


def test_opus_4_5_unsupported_effort_level_translated_to_legacy_thinking():
    """Opus 4.5 accepts output_config.effort but only levels low/medium/high;
    Claude Code defaults to xhigh on newer models, and forwarding that level raw
    would be rejected with "effort='xhigh' is not supported by this model". An
    unsupported level must fall through to the legacy translation (budget-based
    thinking, effort stripped) instead of being preserved."""
    result = _transform("claude-opus-4-5", _claude_code_payload(effort="xhigh", max_tokens=64000))

    assert result["thinking"] == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
    }
    assert "output_config" not in result


def test_opus_4_5_effort_only_unsupported_level_left_for_provider_normalization():
    """An effort-only request (no adaptive thinking) must pass through untouched even
    when the level exceeds what the model supports: provider subclasses own their
    level normalization (bedrock clamps xhigh to the model's ceiling after this base
    transform runs), so consuming the effort here breaks that contract."""
    result = _transform(
        "claude-opus-4-5",
        {"max_tokens": 4096, "output_config": {"effort": "xhigh"}},
    )

    assert result["output_config"] == {"effort": "xhigh"}
    assert "thinking" not in result


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


def test_unrecognized_effort_raises_clean_400():
    """An unrecognized effort value (e.g. a future Anthropic tier) must surface as a
    clean AnthropicError 400, matching _translate_reasoning_effort_to_anthropic,
    rather than leaking litellm's internal BadRequestError."""
    with pytest.raises(AnthropicError) as exc_info:
        _transform("claude-haiku-4-5", _claude_code_payload(effort="turbo"))

    assert exc_info.value.status_code == 400


def test_pinned_temperature_dropped_when_adaptive_downgraded_to_enabled():
    """Regression (#33203): Claude Code's safety classifier sends adaptive thinking +
    temperature=0 to Haiku 4.5. The adaptive interface is downgraded to legacy enabled
    thinking, but Anthropic rejects "temperature may only be set to 1 when thinking is
    enabled". The pinned temperature must be dropped so the request succeeds while the
    downgraded thinking is preserved."""
    params = _claude_code_payload(effort="medium")
    params["temperature"] = 0
    result = _transform("claude-haiku-4-5", params)

    assert result["thinking"] == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    }
    assert "temperature" not in result


def test_temperature_one_preserved_with_enabled_thinking():
    """temperature=1 is compatible with extended thinking, so it must be kept."""
    params = _claude_code_payload(effort="medium")
    params["temperature"] = 1
    result = _transform("claude-haiku-4-5", params)

    assert result["thinking"]["type"] == "enabled"
    assert result["temperature"] == 1


def test_pinned_temperature_preserved_when_thinking_dropped():
    """When thinking is dropped entirely (non-reasoning model), there is no thinking
    conflict, so a pinned temperature must survive untouched."""
    params = _claude_code_payload(effort="medium")
    params["temperature"] = 0
    result = _transform("claude-3-5-haiku-latest", params)

    assert "thinking" not in result
    assert result["temperature"] == 0


def test_pinned_temperature_preserved_for_adaptive_model():
    """Adaptive models (4.6+) own the thinking/temperature relationship natively, so
    the passthrough must not strip a pinned temperature for them."""
    params = _claude_code_payload(effort="high")
    params["temperature"] = 0
    result = _transform("claude-sonnet-4-6", params)

    assert result["thinking"] == {"type": "adaptive"}
    assert result["temperature"] == 0


def test_pinned_temperature_dropped_for_opus_4_5_effort():
    """Opus 4.5 keeps native output_config.effort (extended thinking), which is equally
    incompatible with a pinned non-1 temperature, so the temperature must be dropped."""
    params = _claude_code_payload(effort="medium")
    params["temperature"] = 0
    result = _transform("claude-opus-4-5", params)

    assert result["output_config"] == {"effort": "medium"}
    assert "temperature" not in result


def test_reasoning_effort_budget_capped_below_max_tokens():
    """Sibling defect (#33203): the reasoning_effort alias mapped effort to legacy
    thinking without capping budget_tokens below max_tokens, emitting
    budget_tokens >= max_tokens which Anthropic rejects. A high-effort budget with
    max_tokens=3000 must be capped to 2999."""
    result = _transform(
        "claude-haiku-4-5",
        {"max_tokens": 3000, "reasoning_effort": "high"},
    )

    assert result["thinking"] == {"type": "enabled", "budget_tokens": 2999}


def test_reasoning_effort_thinking_dropped_when_max_tokens_too_small():
    """reasoning_effort thinking is dropped when max_tokens can't fit even the minimum
    budget, so the request still succeeds."""
    result = _transform(
        "claude-haiku-4-5",
        {"max_tokens": 512, "reasoning_effort": "medium"},
    )

    assert "thinking" not in result


def test_reasoning_effort_with_pinned_temperature_drops_temperature():
    """The reasoning_effort alias synthesizes legacy enabled thinking on a non-adaptive
    model; a co-pinned non-1 temperature must be dropped to avoid the Anthropic 400."""
    result = _transform(
        "claude-haiku-4-5",
        {"max_tokens": 8192, "reasoning_effort": "low", "temperature": 0},
    )

    assert result["thinking"] == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    }
    assert "temperature" not in result


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
