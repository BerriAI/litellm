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
from litellm.llms.openai_like.json_loader import SimpleProviderConfig
from litellm.llms.openai_like.messages.transformation import (
    JSONProviderAnthropicMessagesConfig,
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


def test_adaptive_thinking_only_translated_to_legacy_for_haiku_4_5():
    """The minimal autoroute repro: Claude Code sends bare ``thinking={type: adaptive}``
    (no ``output_config``) and the complexity router picks Haiku 4.5, which does not
    support adaptive thinking. Anthropic 400s with "adaptive thinking is not supported on
    this model" unless the flag is dropped, so it must be translated to the legacy extended
    thinking the model does support rather than forwarded raw."""
    result = _transform("claude-haiku-4-5", {"max_tokens": 8192, "thinking": {"type": "adaptive"}})

    assert result["thinking"] == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    }
    assert "output_config" not in result


def test_adaptive_thinking_only_dropped_for_non_reasoning_model():
    """Bare adaptive thinking on a model with no reasoning support at all is silently
    dropped so the request still succeeds instead of being rejected."""
    result = _transform("claude-3-5-haiku-latest", {"max_tokens": 8192, "thinking": {"type": "adaptive"}})

    assert "thinking" not in result


def test_adaptive_thinking_only_preserved_for_4_6():
    """A 4.6+ model natively supports adaptive thinking, so a bare adaptive flag must not
    be rewritten even without output_config."""
    result = _transform("claude-sonnet-4-6", {"max_tokens": 8192, "thinking": {"type": "adaptive"}})

    assert result["thinking"] == {"type": "adaptive"}


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


def test_pinned_temperature_dropped_for_adaptive_model():
    """Adaptive models (4.6+) still 400 on a pinned non-1 temperature: Anthropic's
    restriction applies "when thinking is enabled or in adaptive mode", so the
    passthrough must drop the temperature (keeping the thinking the caller asked
    for) rather than forwarding it untouched."""
    params = _claude_code_payload(effort="high")
    params["temperature"] = 0
    result = _transform("claude-sonnet-4-6", params)

    assert result["thinking"] == {"type": "adaptive"}
    assert "temperature" not in result


def test_pinned_temperature_one_kept_for_adaptive_model():
    """temperature=1 is compatible with adaptive thinking, so it must survive."""
    params = _claude_code_payload(effort="high")
    params["temperature"] = 1
    result = _transform("claude-sonnet-4-6", params)

    assert result["thinking"] == {"type": "adaptive"}
    assert result["temperature"] == 1


def test_low_top_p_dropped_for_adaptive_thinking():
    """Anthropic requires top_p >= 0.95 (or unset) when thinking is active, including
    adaptive mode; a pinned top_p below that must be dropped."""
    params = _claude_code_payload(effort="high")
    params["top_p"] = 0.9
    result = _transform("claude-sonnet-4-6", params)

    assert result["thinking"] == {"type": "adaptive"}
    assert "top_p" not in result


def test_high_top_p_kept_for_adaptive_thinking():
    """top_p >= 0.95 is allowed under thinking, so it must be preserved."""
    params = _claude_code_payload(effort="high")
    params["top_p"] = 0.95
    result = _transform("claude-sonnet-4-6", params)

    assert result["thinking"] == {"type": "adaptive"}
    assert result["top_p"] == 0.95


def test_top_k_dropped_for_adaptive_thinking():
    """top_k must be unset when thinking is active, including adaptive mode."""
    params = _claude_code_payload(effort="high")
    params["top_k"] = 5
    result = _transform("claude-sonnet-4-6", params)

    assert result["thinking"] == {"type": "adaptive"}
    assert "top_k" not in result


def test_sampling_params_kept_when_no_thinking_active():
    """With no thinking at all, sampling params are unrelated and must pass through."""
    result = _transform(
        "claude-sonnet-4-6",
        {"max_tokens": 1024, "temperature": 0, "top_p": 0.9, "top_k": 5},
    )

    assert result["temperature"] == 0
    assert result["top_p"] == 0.9
    assert result["top_k"] == 5


def test_pinned_temperature_dropped_for_opus_4_5_effort():
    """Opus 4.5 keeps native output_config.effort (extended thinking), which is equally
    incompatible with a pinned non-1 temperature, so the temperature must be dropped."""
    params = _claude_code_payload(effort="medium")
    params["temperature"] = 0
    result = _transform("claude-opus-4-5", params)

    assert result["output_config"] == {"effort": "medium"}
    assert "temperature" not in result


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


def test_reasoning_effort_budget_capped_below_max_tokens():
    result = _transform("claude-haiku-4-5", {"max_tokens": 4000, "reasoning_effort": "xhigh"})

    assert result["thinking"] == {"type": "enabled", "budget_tokens": 3999}
    assert result["max_tokens"] == 4000


def test_reasoning_effort_thinking_dropped_when_min_budget_cannot_fit():
    result = _transform("claude-haiku-4-5", {"max_tokens": 1024, "reasoning_effort": "xhigh"})

    assert "thinking" not in result
    assert result["max_tokens"] == 1024


def test_reasoning_effort_budget_capped_for_openai_like_messages_upstream():
    provider = SimpleProviderConfig(
        "meta",
        {
            "base_url": "https://api.meta.ai/v1",
            "api_key_env": "META_API_KEY",
            "supported_endpoints": ["/v1/messages"],
        },
    )

    result = JSONProviderAnthropicMessagesConfig(provider).transform_anthropic_messages_request(
        model="muse-spark-1.2",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 4000, "reasoning_effort": "xhigh"},
        litellm_params={},
        headers={},
    )

    assert result["thinking"] == {"type": "enabled", "budget_tokens": 3999}
    assert result["max_tokens"] == 4000


def test_sampling_params_kept_when_thinking_disabled():
    """An explicit ``thinking={type: disabled}`` is not active thinking, so sampling
    params (even ones Anthropic would reject under active thinking) must pass through."""
    result = _transform(
        "claude-opus-4-6",
        {"max_tokens": 1024, "thinking": {"type": "disabled"}, "temperature": 0, "top_p": 0.9, "top_k": 5},
    )

    assert result["thinking"] == {"type": "disabled"}
    assert result["temperature"] == 0
    assert result["top_p"] == 0.9
    assert result["top_k"] == 5


def test_all_incompatible_sampling_dropped_together_on_anthropic():
    """Combined case: temperature != 1, top_p < 0.95 and top_k all present under active
    adaptive thinking on the first-party Anthropic API must all be dropped together."""
    params = _claude_code_payload(effort="high")
    params.update({"temperature": 0, "top_p": 0.9, "top_k": 5})
    result = _transform("claude-opus-4-6", params)

    assert result["thinking"] == {"type": "adaptive"}
    assert "temperature" not in result
    assert "top_p" not in result
    assert "top_k" not in result


def test_top_p_top_k_not_stripped_on_other_anthropic_compatible_backend():
    """Cross-backend guard: DeepSeek's Anthropic-compatible endpoint has its own sampling
    rules. Temperature is still dropped under active thinking, but the Anthropic-specific
    top_p/top_k stripping must not run there and silently remove user-supplied values."""
    from litellm.llms.deepseek.messages.transformation import (
        DeepSeekAnthropicMessagesConfig,
    )

    params = {
        "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 512},
        "temperature": 0,
        "top_p": 0.9,
        "top_k": 5,
    }
    result = DeepSeekAnthropicMessagesConfig().transform_anthropic_messages_request(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=dict(params),
        litellm_params={},
        headers={},
    )

    assert "temperature" not in result
    assert result["top_p"] == 0.9
    assert result["top_k"] == 5


def test_bedrock_invoke_keeps_pinned_temperature_on_adaptive_effort():
    """Cross-backend regression: ``AmazonAnthropicClaudeMessagesConfig`` inherits this
    transform, and Bedrock Invoke honours a pinned ``temperature`` alongside
    ``output_config.effort`` on an adaptive model. Treating the bare effort level as
    active thinking for every provider silently dropped that temperature, which broke
    ``test_bedrock_messages_allowlist_filters_anthropic_only_fields``."""
    from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
        AmazonAnthropicClaudeMessagesConfig,
    )
    from litellm.types.router import GenericLiteLLMParams

    params = {"max_tokens": 4096, "temperature": 0.5, "output_config": {"effort": "low"}}
    result = AmazonAnthropicClaudeMessagesConfig().transform_anthropic_messages_request(
        model="anthropic.claude-opus-4-7",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=dict(params),
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert result["temperature"] == 0.5


def test_bedrock_keeps_pinned_temperature_for_claude_code_adaptive_effort_shape():
    """The shape Claude Code actually sends -- an adaptive ``thinking`` block *and* an
    ``output_config.effort`` level -- on an adaptive model, where the earlier passes
    leave both keys in place. The bare ``effort_set`` arm would still classify this as
    active thinking, so the provider early return is what keeps Bedrock's pinned
    temperature: only the first-party API rejects it under adaptive thinking."""
    from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
        AmazonAnthropicClaudeMessagesConfig,
    )
    from litellm.types.router import GenericLiteLLMParams

    params = _claude_code_payload(effort="high")
    params["temperature"] = 0.5
    result = AmazonAnthropicClaudeMessagesConfig().transform_anthropic_messages_request(
        model="anthropic.claude-opus-4-7",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params=params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert result["thinking"] == {"type": "adaptive"}
    assert result["temperature"] == 0.5


def test_adaptive_block_is_not_active_thinking_on_other_backends():
    """A bare ``thinking={type: adaptive}`` says nothing about a backend's sampling rules.

    This is asserted at the helper level on a non-adaptive model because the earlier
    passes hide the case end to end: adaptive models are handled by the provider early
    return, and for everyone else the block is either translated to a legacy ``enabled``
    budget or dropped outright. So a direct call is the only place the ``first_party``
    qualifier on ``adaptive`` is observable, and it has to hold -- treating the block as
    active thinking for every provider is what silently dropped Bedrock's pinned
    temperature in the first place.
    """
    params = {
        "max_tokens": 4096,
        "temperature": 0.5,
        "top_p": 0.5,
        "top_k": 5,
        "thinking": {"type": "adaptive"},
    }

    for provider in ("bedrock", "vertex_ai", "azure", "deepseek"):
        candidate = dict(params)
        AnthropicMessagesConfig._drop_incompatible_sampling_for_thinking("claude-opus-4-5", candidate, provider)
        assert candidate == params, f"{provider} stripped sampling params for a non-first-party adaptive block"


def test_non_numeric_top_p_forwarded_under_thinking():
    """A non-numeric top_p (e.g. a serialized string from an upstream gateway) must not
    raise a TypeError during the comparison; it is left for Anthropic to validate."""
    params = _claude_code_payload(effort="high")
    params["top_p"] = "0.9"
    result = _transform("claude-opus-4-6", params)

    assert result["thinking"] == {"type": "adaptive"}
    assert result["top_p"] == "0.9"
