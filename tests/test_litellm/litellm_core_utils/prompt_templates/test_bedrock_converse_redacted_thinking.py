"""Regression tests for Bedrock Converse redacted_thinking block preservation.

See BerriAI/litellm#43009:
With a Bedrock Converse model (e.g. Claude 3.7 Sonnet / Claude Sonnet 4),
redacted thinking returned by the model must be preserved when replaying the
conversation back. Dropping or altering redacted_thinking blocks causes AWS Bedrock
to reject subsequent requests with "messages.1.content.0: Invalid signature in thinking block".
"""

import pytest

from litellm.litellm_core_utils.prompt_templates.factory import (
    BedrockConverseMessagesProcessor,
    _bedrock_converse_messages_pt,
)


def test_bedrock_converse_preserves_redacted_thinking_in_content():
    """Verify redacted_thinking inside assistant content list is transformed to reasoningContent.redactedContent."""
    messages = [
        {"role": "user", "content": "What is 2 + 2?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Calculating 2 + 2",
                    "signature": "sig_valid_123",
                },
                {
                    "type": "redacted_thinking",
                    "data": "opaque_redacted_data_token_abc",
                },
                {"type": "text", "text": "4"},
            ],
        },
    ]

    result = _bedrock_converse_messages_pt(
        messages=messages,
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        llm_provider="bedrock_converse",
    )

    assert len(result) == 2
    assistant_blocks = result[1]["content"]
    assert len(assistant_blocks) == 3

    # First block: signed reasoningText
    assert "reasoningContent" in assistant_blocks[0]
    assert assistant_blocks[0]["reasoningContent"]["reasoningText"] == {
        "text": "Calculating 2 + 2",
        "signature": "sig_valid_123",
    }

    # Second block: redactedContent
    assert "reasoningContent" in assistant_blocks[1]
    assert assistant_blocks[1]["reasoningContent"]["redactedContent"] == "opaque_redacted_data_token_abc"

    # Third block: text answer
    assert assistant_blocks[2] == {"text": "4"}


def test_bedrock_converse_preserves_redacted_thinking_in_thinking_blocks_attribute():
    """Verify redacted_thinking in assistant message thinking_blocks attribute is preserved."""
    messages = [
        {"role": "user", "content": "Evaluate logic puzzle"},
        {
            "role": "assistant",
            "content": "The answer is 42",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Step 1 reasoning",
                    "signature": "sig_step_1",
                },
                {
                    "type": "redacted_thinking",
                    "data": "opaque_redacted_step_2",
                },
            ],
        },
    ]

    result = _bedrock_converse_messages_pt(
        messages=messages,
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        llm_provider="bedrock_converse",
    )

    assert len(result) == 2
    assistant_blocks = result[1]["content"]
    assert len(assistant_blocks) == 3

    assert "reasoningContent" in assistant_blocks[0]
    assert assistant_blocks[0]["reasoningContent"]["reasoningText"]["signature"] == "sig_step_1"

    assert "reasoningContent" in assistant_blocks[1]
    assert assistant_blocks[1]["reasoningContent"]["redactedContent"] == "opaque_redacted_step_2"

    assert assistant_blocks[2] == {"text": "The answer is 42"}


@pytest.mark.asyncio
async def test_bedrock_converse_async_matches_sync_for_redacted_thinking():
    """Verify async and sync Bedrock Converse message transformation yield identical output for redacted_thinking."""
    messages = [
        {"role": "user", "content": "Run tool"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "planning",
                    "signature": "sig_plan",
                },
                {
                    "type": "redacted_thinking",
                    "data": "redacted_binary_blob",
                },
                {
                    "type": "text",
                    "text": "calling tool",
                },
            ],
        },
    ]

    sync_result = _bedrock_converse_messages_pt(
        messages=messages,
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        llm_provider="bedrock_converse",
    )

    async_result = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages=messages,
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        llm_provider="bedrock_converse",
    )

    assert sync_result == async_result
    assert async_result[1]["content"][1]["reasoningContent"]["redactedContent"] == "redacted_binary_blob"


def test_bedrock_converse_multi_turn_tool_loop_with_redacted_thinking():
    """Verify multi-turn tool replay preserves redacted_thinking in sequence with toolUse."""
    messages = [
        {"role": "user", "content": "Fetch data"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let's call the API", "signature": "sig_api_call"},
                {"type": "redacted_thinking", "data": "redacted_safety_check_hash"},
            ],
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "fetch_api", "arguments": '{"query": "data"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"status": "ok"}',
        },
        {"role": "user", "content": "continue"},
    ]

    result = _bedrock_converse_messages_pt(
        messages=messages,
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        llm_provider="bedrock_converse",
    )

    # Turn 0: user
    assert result[0]["role"] == "user"
    # Turn 1: assistant with reasoningText, redactedContent, and toolUse
    assert result[1]["role"] == "assistant"
    assistant_blocks = result[1]["content"]
    assert any(
        b.get("reasoningContent", {}).get("redactedContent") == "redacted_safety_check_hash" for b in assistant_blocks
    )
    assert any("toolUse" in b for b in assistant_blocks)
