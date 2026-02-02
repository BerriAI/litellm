"""
Test that Bedrock streaming responses always use choice index 0,
regardless of contentBlockIndex value.

Bedrock's contentBlockIndex identifies content blocks within a message (e.g.,
text=0, toolUse=1), NOT parallel completions. Since Bedrock doesn't support
n > 1, all chunks must use choice index 0.

References:
- Bedrock InferenceConfiguration (no n parameter):
  https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InferenceConfiguration.html
- OpenAI choice.index (for n > 1):
  https://platform.openai.com/docs/api-reference/chat/object
"""

from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder


class TestBedrockStreamingChoiceIndex:
    """Test that all streaming chunks use choice index 0."""

    def test_tool_call_chunk_uses_choice_index_zero(self):
        """
        Core regression test: tool call chunks must use choice index 0,
        not contentBlockIndex (which is 1 for tool calls).

        This was the bug - contentBlockIndex was incorrectly used as choice.index,
        breaking OpenAI SDK's ChatCompletionAccumulator.
        """
        handler = AWSEventStreamDecoder(model="anthropic.claude-3-sonnet-20240229-v1:0")

        # First, simulate a tool use start event on contentBlockIndex 1
        start_chunk = {
            "start": {
                "toolUse": {
                    "toolUseId": "tooluse_abc123",
                    "name": "get_weather",
                }
            },
            "contentBlockIndex": 1,  # Tool calls are on index 1
        }

        start_result = handler.converse_chunk_parser(start_chunk)

        # Choice index should be 0, NOT contentBlockIndex (1)
        assert start_result.choices[0].index == 0
        assert start_result.choices[0].delta.tool_calls is not None
        assert start_result.choices[0].delta.tool_calls[0]["id"] == "tooluse_abc123"

        # Now simulate tool use delta on contentBlockIndex 1
        delta_chunk = {
            "delta": {
                "toolUse": {
                    "input": '{"location": "San Francisco"}'
                }
            },
            "contentBlockIndex": 1,  # Tool calls are on index 1
        }

        delta_result = handler.converse_chunk_parser(delta_chunk)

        # Choice index should still be 0, NOT contentBlockIndex (1)
        assert delta_result.choices[0].index == 0
        assert delta_result.choices[0].delta.tool_calls is not None
        assert delta_result.choices[0].delta.tool_calls[0]["function"]["arguments"] == '{"location": "San Francisco"}'

    def test_mixed_content_blocks_all_use_choice_index_zero(self):
        """
        Integration test simulating a realistic streaming session:
        text (contentBlockIndex=0) → tool call (contentBlockIndex=1) → finish.

        All chunks must have choice.index=0 for OpenAI SDK compatibility.
        """
        handler = AWSEventStreamDecoder(model="anthropic.claude-3-sonnet-20240229-v1:0")

        # Chunk 1: Text on contentBlockIndex 0
        text_chunk = {
            "delta": {"text": "Let me check the weather."},
            "contentBlockIndex": 0,
        }
        result1 = handler.converse_chunk_parser(text_chunk)
        assert result1.choices[0].index == 0, "Text chunk should have index=0"

        # Chunk 2: Tool call start on contentBlockIndex 1
        tool_start_chunk = {
            "start": {
                "toolUse": {
                    "toolUseId": "tool_xyz",
                    "name": "get_weather",
                }
            },
            "contentBlockIndex": 1,
        }
        result2 = handler.converse_chunk_parser(tool_start_chunk)
        assert result2.choices[0].index == 0, "Tool start should have index=0, not contentBlockIndex=1"

        # Chunk 3: Tool call delta on contentBlockIndex 1
        tool_delta_chunk = {
            "delta": {
                "toolUse": {
                    "input": '{"city": "NYC"}'
                }
            },
            "contentBlockIndex": 1,
        }
        result3 = handler.converse_chunk_parser(tool_delta_chunk)
        assert result3.choices[0].index == 0, "Tool delta should have index=0, not contentBlockIndex=1"

        # Chunk 4: Finish reason
        finish_chunk = {
            "stopReason": "tool_use",
        }
        result4 = handler.converse_chunk_parser(finish_chunk)
        assert result4.choices[0].index == 0, "Finish reason should have index=0"
