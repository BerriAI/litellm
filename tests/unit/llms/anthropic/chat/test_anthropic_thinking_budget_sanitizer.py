"""
Tests for Anthropic thinking budget vs. max_tokens sanitization.

Anthropic API requires max_tokens > budget_tokens (and budget_tokens >= 1024).
When budget_tokens >= max_tokens, LiteLLM must cap budget_tokens to max_tokens - 1.
When max_tokens <= 1024 and drop_params=True, thinking should be dropped to prevent 400 errors.
"""

from litellm.llms.anthropic.chat.transformation import AnthropicConfig


def test_thinking_budget_capped_below_max_tokens_in_map_openai_params():
    """When budget_tokens > max_tokens, budget_tokens is capped to max_tokens - 1."""
    config = AnthropicConfig()
    result = config.map_openai_params(
        non_default_params={
            "thinking": {"type": "enabled", "budget_tokens": 4000},
            "max_tokens": 3000,
        },
        optional_params={},
        model="claude-3-7-sonnet-20250219",
        drop_params=True,
    )
    assert "thinking" in result
    assert result["thinking"]["type"] == "enabled"
    assert result["thinking"]["budget_tokens"] == 2999
    assert result["max_tokens"] == 3000


def test_thinking_budget_equal_max_tokens_capped_in_map_openai_params():
    """When budget_tokens == max_tokens, budget_tokens is capped to max_tokens - 1."""
    config = AnthropicConfig()
    result = config.map_openai_params(
        non_default_params={
            "thinking": {"type": "enabled", "budget_tokens": 4000},
            "max_tokens": 4000,
        },
        optional_params={},
        model="claude-3-7-sonnet-20250219",
        drop_params=True,
    )
    assert "thinking" in result
    assert result["thinking"]["type"] == "enabled"
    assert result["thinking"]["budget_tokens"] == 3999
    assert result["max_tokens"] == 4000


def test_thinking_budget_untouched_when_max_tokens_is_greater():
    """When budget_tokens < max_tokens, budget_tokens is preserved as-is."""
    config = AnthropicConfig()
    result = config.map_openai_params(
        non_default_params={
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "max_tokens": 4096,
        },
        optional_params={},
        model="claude-3-7-sonnet-20250219",
        drop_params=True,
    )
    assert "thinking" in result
    assert result["thinking"]["type"] == "enabled"
    assert result["thinking"]["budget_tokens"] == 2048
    assert result["max_tokens"] == 4096


def test_thinking_dropped_when_max_tokens_below_minimum_with_drop_params():
    """When max_tokens <= 1024 and drop_params=True, thinking is dropped."""
    config = AnthropicConfig()
    result = config.map_openai_params(
        non_default_params={
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "max_tokens": 512,
        },
        optional_params={},
        model="claude-3-7-sonnet-20250219",
        drop_params=True,
    )
    assert "thinking" not in result
    assert result["max_tokens"] == 512


def test_thinking_kept_when_max_tokens_below_minimum_without_drop_params():
    """When drop_params=False, thinking is preserved even if max_tokens <= minimum."""
    config = AnthropicConfig()
    result = config.map_openai_params(
        non_default_params={
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "max_tokens": 512,
        },
        optional_params={},
        model="claude-3-7-sonnet-20250219",
        drop_params=False,
    )
    assert "thinking" in result
    assert result["thinking"]["budget_tokens"] == 2048


def test_thinking_budget_capped_in_transform_request():
    """Sanitizer runs at the transform_request chokepoint as well."""
    config = AnthropicConfig()
    data = config.transform_request(
        model="claude-3-7-sonnet-20250219",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={
            "thinking": {"type": "enabled", "budget_tokens": 5000},
            "max_tokens": 4500,
        },
        litellm_params={},
        headers={},
    )
    assert data["thinking"]["type"] == "enabled"
    assert data["thinking"]["budget_tokens"] == 4499
    assert data["max_tokens"] == 4500
