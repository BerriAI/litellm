"""
Tests for Bedrock Converse thinking budget vs. max_tokens sanitization.

AWS Bedrock Converse API requires maxTokens > budget_tokens (and budget_tokens >= 1024).
When budget_tokens >= maxTokens, LiteLLM must cap budget_tokens to maxTokens - 1.
When maxTokens <= 1024 and drop_params=True:
- If conversation history contains thinking blocks, thinking must remain enabled.
- If conversation history has no thinking blocks, thinking should be dropped to prevent 400 errors.
Caller parameter dictionaries must NEVER be modified in place when capping or clamping budgets.
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


def test_bedrock_capping_does_not_modify_caller_owned_dict():
    """Capping budget_tokens must not mutate the caller's parameter dict."""
    config = AmazonConverseConfig()
    caller_thinking = {"type": "enabled", "budget_tokens": 4000}
    result = config.map_openai_params(
        non_default_params={
            "thinking": caller_thinking,
            "max_tokens": 3000,
        },
        optional_params={},
        model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        drop_params=True,
    )
    assert result["thinking"]["budget_tokens"] == 2999
    # Verify the original caller dict is unchanged
    assert caller_thinking["budget_tokens"] == 4000


def test_bedrock_clamping_does_not_modify_caller_owned_dict():
    """Clamping budget_tokens < 1024 must not mutate the caller's parameter dict."""
    config = AmazonConverseConfig()
    caller_thinking = {"type": "enabled", "budget_tokens": 500}
    result = config.map_openai_params(
        non_default_params={
            "thinking": caller_thinking,
            "max_tokens": 2000,
        },
        optional_params={},
        model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        drop_params=True,
    )
    assert result["thinking"]["budget_tokens"] == 1024
    # Verify the original caller dict is unchanged
    assert caller_thinking["budget_tokens"] == 500


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


def test_bedrock_thinking_dropped_in_transform_request_when_no_thinking_history():
    """When maxTokens <= 1024 and drop_params=True without thinking history, thinking is dropped."""
    config = AmazonConverseConfig()
    data = config.transform_request(
        model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "maxTokens": 512,
        },
        litellm_params={"drop_params": True},
        headers={},
    )
    # In Bedrock converse, additionalModelRequestFields holds thinking
    additional = data.get("additionalModelRequestFields", {})
    assert "thinking" not in additional
    assert data["inferenceConfig"]["maxTokens"] == 512


def test_bedrock_thinking_preserved_in_transform_request_when_thinking_history_present():
    """When maxTokens <= 1024 and drop_params=True with thinking history, thinking is preserved."""
    config = AmazonConverseConfig()
    data = config.transform_request(
        model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Analyzing reasoning path...",
                        "signature": "valid_sig",
                    },
                    {"type": "text", "text": "Result"},
                ],
                "thinking_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "Analyzing reasoning path...",
                        "signature": "valid_sig",
                    }
                ],
            },
            {"role": "user", "content": "Follow-up question"},
        ],
        optional_params={
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "maxTokens": 512,
        },
        litellm_params={"drop_params": True},
        headers={},
    )
    additional = data.get("additionalModelRequestFields", {})
    assert "thinking" in additional
    assert additional["thinking"]["type"] == "enabled"
    assert data["inferenceConfig"]["maxTokens"] == 512


def test_bedrock_thinking_kept_when_max_tokens_below_minimum_without_drop_params():
    """When drop_params=False, thinking is preserved even if maxTokens <= minimum."""
    config = AmazonConverseConfig()
    data = config.transform_request(
        model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "maxTokens": 512,
        },
        litellm_params={"drop_params": False},
        headers={},
    )
    additional = data.get("additionalModelRequestFields", {})
    assert "thinking" in additional
    assert data["inferenceConfig"]["maxTokens"] == 512
