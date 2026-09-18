"""Verify DeepSeek `reasoning_effort` passes through the transformation.

Covers the fix that preserves the effort value (low/medium/high/xhigh/max) into
the request body when thinking is on, instead of mapping it to a binary thinking
toggle and discarding it. See https://api-docs.deepseek.com/guides/thinking_mode
"""
from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


def _map(non_default_params, drop_params=True):
    cfg = DeepSeekChatConfig()
    return cfg.map_openai_params(non_default_params, {}, "deepseek/deepseek-v4-flash", drop_params)


def test_reasoning_effort_low_passes_through():
    out = _map({"reasoning_effort": "low"})
    assert out["reasoning_effort"] == "low"
    assert out["thinking"] == {"type": "enabled"}


def test_reasoning_effort_max_passes_through():
    out = _map({"reasoning_effort": "max"})
    assert out["reasoning_effort"] == "max"
    assert out["thinking"] == {"type": "enabled"}


def test_reasoning_effort_none_disables_thinking():
    out = _map({"reasoning_effort": "none"})
    assert out["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in out


def test_explicit_thinking_disabled_wins_and_drops_effort():
    out = _map({"thinking": {"type": "disabled"}, "reasoning_effort": "low"})
    assert out["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in out


def test_no_reasoning_effort_leaves_no_key():
    out = _map({})
    assert out.get("reasoning_effort") is None


def test_explicit_enabled_thinking_with_none_effort_not_contradictory():
    # "none" is a thinking-OFF sentinel, not a real effort value: it must not be
    # passed alongside explicit thinking:enabled (would contradict the provider).
    out = _map({"thinking": {"type": "enabled"}, "reasoning_effort": "none"})
    assert out["thinking"] == {"type": "enabled"}
    assert "reasoning_effort" not in out
