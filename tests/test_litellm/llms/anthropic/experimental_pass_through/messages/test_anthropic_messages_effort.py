import litellm
import pytest
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


def test_effort_dropped_for_non_reasoning_model_with_drop_params(monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", True)
    result = _transform("claude-3-5-haiku-latest", _claude_code_payload(effort="medium"))

    assert "thinking" not in result
    assert "output_config" not in result


def test_effort_raises_for_non_reasoning_model_without_drop_params(monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", False)

    with pytest.raises(Exception, match="drop_params"):
        _transform("claude-3-5-haiku-latest", _claude_code_payload(effort="medium"))


def test_drop_params_from_litellm_params_gates_the_drop(monkeypatch):
    """A per-request drop_params (threaded via litellm_params) must also permit the
    drop even when the global litellm.drop_params is False."""
    monkeypatch.setattr(litellm, "drop_params", False)
    result = _transform(
        "claude-3-5-haiku-latest",
        _claude_code_payload(effort="medium"),
        litellm_params={"drop_params": True},
    )

    assert "thinking" not in result
    assert "output_config" not in result


def test_effort_translated_but_residual_output_config_dropped_for_haiku_4_5(monkeypatch):
    """output_config may carry `format` (structured outputs) alongside effort. On a
    model without output_config support, effort is translated to thinking and the
    residual output_config is dropped under drop_params."""
    monkeypatch.setattr(litellm, "drop_params", True)
    result = _transform(
        "claude-haiku-4-5",
        _claude_code_payload(effort="medium", format={"type": "json_schema"}),
    )

    assert result["thinking"] == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    }
    assert "output_config" not in result


def test_budget_capped_below_max_tokens():
    """Adaptive thinking carries no budget, so the translated legacy budget must be
    capped below max_tokens (Anthropic requires max_tokens > budget_tokens). A
    high-effort budget (4096) with max_tokens=3000 must be capped to 2999."""
    result = _transform(
        "claude-haiku-4-5", _claude_code_payload(effort="high", max_tokens=3000)
    )

    assert result["thinking"] == {"type": "enabled", "budget_tokens": 2999}


def test_thinking_dropped_when_max_tokens_too_small_for_min_budget(monkeypatch):
    """When max_tokens can't fit even the minimum thinking budget, thinking is
    dropped so the request still succeeds rather than being rejected."""
    monkeypatch.setattr(litellm, "drop_params", True)
    result = _transform(
        "claude-haiku-4-5", _claude_code_payload(effort="medium", max_tokens=512)
    )

    assert "thinking" not in result
    assert "output_config" not in result
