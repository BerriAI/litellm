"""
Test that Bedrock streaming responses always use choice index 0,
regardless of contentBlockIndex value.

Bedrock's contentBlockIndex identifies content blocks within a message (e.g.,
text=0, toolUse=1), NOT parallel completions. Since Bedrock doesn't support
n > 1, all chunks must use choice index 0.

This includes extended-thinking responses where the reasoning block arrives on
contentBlockIndex=0 and the text block on contentBlockIndex=1 — both must be
normalised to choices[0].index == 0 for OpenAI-SDK compatibility.
(Regression for https://github.com/BerriAI/litellm/issues/23178)

References:
- Bedrock InferenceConfiguration (no n parameter):
  https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InferenceConfiguration.html
- OpenAI choice.index (for n > 1):
  https://platform.openai.com/docs/api-reference/chat/object
"""

from litellm.llms.bedrock.chat.invoke_handler import (
    AWSEventStreamDecoder,
    AmazonAnthropicClaudeStreamDecoder,
)


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

    # ------------------------------------------------------------------
    # Extended-thinking / reasoning tests (issue #23178)
    # ------------------------------------------------------------------

    def test_thinking_block_chunks_use_choice_index_zero(self):
        """
        Regression for https://github.com/BerriAI/litellm/issues/23178.

        When Claude extended-thinking is enabled the Bedrock converse API emits:
          contentBlockIndex=0  ->  reasoning / thinking block
          contentBlockIndex=1  ->  text block

        LiteLLM must normalise both to choices[0].index == 0 so that
        OpenAI-compatible clients (e.g. openai.ChatCompletionAccumulator) do not
        crash when they see an unexpected index switch mid-stream.
        """
        handler = AWSEventStreamDecoder(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0"
        )

        # thinking block start (contentBlockIndex=0)
        r = handler.converse_chunk_parser(
            {"start": {"reasoningContent": {}}, "contentBlockIndex": 0}
        )
        assert r.choices[0].index == 0, "thinking start should have index=0"

        # thinking block delta (contentBlockIndex=0)
        r = handler.converse_chunk_parser(
            {
                "delta": {"reasoningContent": {"text": "Let me think step by step..."}},
                "contentBlockIndex": 0,
            }
        )
        assert r.choices[0].index == 0, "thinking delta should have index=0"
        assert r.choices[0].delta.reasoning_content == "Let me think step by step..."

        # thinking block stop (contentBlockIndex=0)
        r = handler.converse_chunk_parser({"contentBlockIndex": 0})
        assert r.choices[0].index == 0, "thinking stop should have index=0"

        # text block start (contentBlockIndex=1) — this is the previously broken case
        r = handler.converse_chunk_parser(
            {"start": {}, "contentBlockIndex": 1}
        )
        assert r.choices[0].index == 0, (
            "text start after thinking block should have index=0, not contentBlockIndex=1"
        )

        # text block delta (contentBlockIndex=1)
        r = handler.converse_chunk_parser(
            {"delta": {"text": "The answer is 42."}, "contentBlockIndex": 1}
        )
        assert r.choices[0].index == 0, (
            "text delta after thinking block should have index=0, not contentBlockIndex=1"
        )
        assert r.choices[0].delta.content == "The answer is 42."

        # text block stop (contentBlockIndex=1)
        r = handler.converse_chunk_parser({"contentBlockIndex": 1})
        assert r.choices[0].index == 0, "text stop should have index=0"

        # message stop
        r = handler.converse_chunk_parser({"stopReason": "end_turn"})
        assert r.choices[0].index == 0, "message stop should have index=0"

    def test_thinking_signature_chunk_uses_choice_index_zero(self):
        """The reasoning block ends with a cryptographic signature chunk; it must also carry index=0."""
        handler = AWSEventStreamDecoder(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        r = handler.converse_chunk_parser(
            {
                "delta": {"reasoningContent": {"signature": "abc123signatureXYZ"}},
                "contentBlockIndex": 0,
            }
        )
        assert r.choices[0].index == 0, "signature delta should have index=0"
        assert r.choices[0].delta.thinking_blocks is not None
        assert len(r.choices[0].delta.thinking_blocks) > 0

    def test_anthropic_invoke_decoder_thinking_uses_choice_index_zero(self):
        """
        When Claude models are invoked via the Bedrock /invoke endpoint
        (bedrock_invoke_provider='anthropic'), AmazonAnthropicClaudeStreamDecoder
        is used.  It must normalise the raw Anthropic index=1 on text blocks to
        choices[0].index == 0, matching the behaviour of the direct Anthropic provider.
        """
        handler = AmazonAnthropicClaudeStreamDecoder(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            sync_stream=True,
        )

        anthropic_chunks = [
            # thinking block — raw Anthropic index=0
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "I should multiply."},
            },
            {"type": "content_block_stop", "index": 0},
            # text block — raw Anthropic index=1 (the problematic case)
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "27 x 453 = 12231"},
            },
            {"type": "content_block_stop", "index": 1},
        ]

        for chunk in anthropic_chunks:
            result = handler._chunk_parser(chunk)
            assert result.choices[0].index == 0, (
                f"chunk type={chunk.get('type')} index={chunk.get('index')} "
                f"produced choices[0].index={result.choices[0].index}, expected 0"
            )
