"""
Tests for Bedrock Converse thinking budget vs. max_tokens sanitization.

AWS Bedrock Converse API requires maxTokens > budget_tokens (and budget_tokens >= 1024).
When budget_tokens >= maxTokens, LiteLLM must cap budget_tokens to maxTokens - 1.
When maxTokens <= 1024 and drop_params=True, thinking should be dropped to prevent 400 errors.
When drop_params=False, thinking is preserved.
"""

from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig


def test_bedrock_thinking_budget_capped_below_max_tokens_in_map_openai_params():
    """When budget_tokens > maxTokens, budget_tokens is capped to maxTokens - 1."""
    config = AmazonConverseConfig()
    result = config.map_openai_params(
        non_default_params={
            "thinking": {"type": "enabled", "budget_tokens": 4000},
            "max_tokens": 3000,
        },
        optional_params={},
        model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        drop_params=True,
    )
    assert "thinking" in result
    assert result["thinking"]["type"] == "enabled"
    assert result["thinking"]["budget_tokens"] == 2999
    assert result["maxTokens"] == 3000


def test_bedrock_thinking_budget_equal_max_tokens_capped_in_map_openai_params():
    """When budget_tokens == maxTokens, budget_tokens is capped to maxTokens - 1."""
    config = AmazonConverseConfig()
    result = config.map_openai_params(
        non_default_params={
            "thinking": {"type": "enabled", "budget_tokens": 4000},
            "max_tokens": 4000,
        },
        optional_params={},
        model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        drop_params=True,
    )
    assert "thinking" in result
    assert result["thinking"]["type"] == "enabled"
    assert result["thinking"]["budget_tokens"] == 3999
    assert result["maxTokens"] == 4000


def test_bedrock_thinking_budget_untouched_when_max_tokens_is_greater():
    """When budget_tokens < maxTokens, budget_tokens is preserved as-is."""
    config = AmazonConverseConfig()
    result = config.map_openai_params(
        non_default_params={
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "max_tokens": 4096,
        },
        optional_params={},
        model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        drop_params=True,
    )
    assert "thinking" in result
    assert result["thinking"]["type"] == "enabled"
    assert result["thinking"]["budget_tokens"] == 2048
    assert result["maxTokens"] == 4096


def test_bedrock_thinking_dropped_when_max_tokens_below_minimum_with_drop_params():
    """When maxTokens <= 1024 and drop_params=True, thinking is dropped."""
    config = AmazonConverseConfig()
    result = config.map_openai_params(
        non_default_params={
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "max_tokens": 512,
        },
        optional_params={},
        model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        drop_params=True,
    )
    assert "thinking" not in result
    assert result["maxTokens"] == 512


def test_bedrock_thinking_kept_when_max_tokens_below_minimum_without_drop_params():
    """When drop_params=False, thinking is preserved even if maxTokens <= minimum."""
    config = AmazonConverseConfig()
    result = config.map_openai_params(
        non_default_params={
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "max_tokens": 512,
        },
        optional_params={},
        model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        drop_params=False,
    )
    assert "thinking" in result
    assert result["thinking"]["budget_tokens"] == 2048
    assert result["maxTokens"] == 512


def test_bedrock_thinking_budget_clamped_to_minimum_when_below_1024():
    """When budget_tokens < 1024 and maxTokens > 1024, budget is clamped to 1024."""
    config = AmazonConverseConfig()
    result = config.map_openai_params(
        non_default_params={
            "thinking": {"type": "enabled", "budget_tokens": 500},
            "max_tokens": 2000,
        },
        optional_params={},
        model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        drop_params=True,
    )
    assert "thinking" in result
    assert result["thinking"]["budget_tokens"] == 1024
    assert result["maxTokens"] == 2000
