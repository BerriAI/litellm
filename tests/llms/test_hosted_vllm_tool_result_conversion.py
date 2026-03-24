"""
Tests for Anthropic tool_result → OpenAI tool message conversion in hosted_vllm.

When Claude Code uses tools like WebFetch via LiteLLM routing to hosted_vllm,
tool results arrive as Anthropic-format tool_result content blocks inside user
messages. These must be converted to OpenAI-format tool role messages.

Fixes: https://github.com/BerriAI/litellm/issues/24491
"""

import pytest

from litellm.llms.hosted_vllm.chat.transformation import HostedVLLMChatConfig


@pytest.fixture
def config():
    return HostedVLLMChatConfig()


class TestExtractToolResultContent:
    """Unit tests for _extract_tool_result_content helper."""

    def test_string_content(self):
        assert HostedVLLMChatConfig._extract_tool_result_content("hello") == "hello"

    def test_none_content(self):
        assert HostedVLLMChatConfig._extract_tool_result_content(None) == ""

    def test_list_with_text_blocks(self):
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert HostedVLLMChatConfig._extract_tool_result_content(content) == "first second"

    def test_list_with_single_text_block(self):
        content = [{"type": "text", "text": "fetched content"}]
        assert HostedVLLMChatConfig._extract_tool_result_content(content) == "fetched content"

    def test_list_with_strings(self):
        content = ["part1", "part2"]
        assert HostedVLLMChatConfig._extract_tool_result_content(content) == "part1 part2"

    def test_empty_list(self):
        assert HostedVLLMChatConfig._extract_tool_result_content([]) == ""

    def test_empty_string(self):
        assert HostedVLLMChatConfig._extract_tool_result_content("") == ""


class TestToolResultConversion:
    """Test that tool_result blocks in user messages are converted to OpenAI tool messages."""

    def test_single_tool_result_string_content(self, config):
        """Basic case: single tool_result with string content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc123",
                        "content": "fetched content from web",
                    }
                ],
            }
        ]
        result = config._transform_messages(messages, model="test-model")
        # Should produce a single tool message (user message with no remaining content is dropped)
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "toolu_abc123"
        assert tool_msgs[0]["content"] == "fetched content from web"

    def test_single_tool_result_list_content(self, config):
        """tool_result with list content containing text blocks."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_xyz",
                        "content": [
                            {"type": "text", "text": "page title"},
                            {"type": "text", "text": "page body"},
                        ],
                    }
                ],
            }
        ]
        result = config._transform_messages(messages, model="test-model")
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "toolu_xyz"
        assert tool_msgs[0]["content"] == "page title page body"

    def test_tool_result_with_no_content(self, config):
        """tool_result with missing content field."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_empty",
                    }
                ],
            }
        ]
        result = config._transform_messages(messages, model="test-model")
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "toolu_empty"
        assert tool_msgs[0]["content"] == ""

    def test_tool_result_mixed_with_text(self, config):
        """User message with both text and tool_result blocks."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here are the results:"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_fetch1",
                        "content": "fetched data",
                    },
                ],
            }
        ]
        result = config._transform_messages(messages, model="test-model")

        # Should have tool message first, then user message with remaining text
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        user_msgs = [m for m in result if m.get("role") == "user"]

        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "fetched data"

        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == [{"type": "text", "text": "Here are the results:"}]

    def test_multiple_tool_results(self, config):
        """Multiple tool_result blocks in a single user message."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "result 1",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_2",
                        "content": "result 2",
                    },
                ],
            }
        ]
        result = config._transform_messages(messages, model="test-model")
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == "toolu_1"
        assert tool_msgs[0]["content"] == "result 1"
        assert tool_msgs[1]["tool_call_id"] == "toolu_2"
        assert tool_msgs[1]["content"] == "result 2"

    def test_no_tool_results_unchanged(self, config):
        """Messages without tool_result blocks should not be modified."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": [{"type": "text", "text": "How are you?"}]},
        ]
        result = config._transform_messages(messages, model="test-model")
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"

    def test_preserves_message_order(self, config):
        """Verify tool messages are inserted before the user message they came from."""
        messages = [
            {"role": "assistant", "content": "I'll fetch that for you."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_web",
                        "content": "web page content",
                    },
                    {"type": "text", "text": "Please summarize this."},
                ],
            },
        ]
        result = config._transform_messages(messages, model="test-model")
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[1]["content"] == "web page content"
        assert result[2]["role"] == "user"
        assert result[2]["content"] == [{"type": "text", "text": "Please summarize this."}]

    def test_claude_code_webfetch_scenario(self, config):
        """
        End-to-end scenario: Claude Code uses WebFetch tool, result comes back
        as Anthropic tool_result, needs conversion for hosted_vllm.
        """
        messages = [
            {"role": "user", "content": "What's on example.com?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll fetch that page for you."},
                ],
                "tool_calls": [
                    {
                        "id": "call_webfetch_001",
                        "type": "function",
                        "function": {
                            "name": "WebFetch",
                            "arguments": '{"url": "https://example.com"}',
                        },
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_webfetch_001",
                        "content": "<html><body>Example Domain</body></html>",
                    }
                ],
            },
        ]
        result = config._transform_messages(messages, model="test-model")

        # The tool_result should be converted to a tool message
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_webfetch_001"
        assert "Example Domain" in tool_msgs[0]["content"]

        # Original user and assistant messages should be preserved
        user_msgs = [m for m in result if m.get("role") == "user"]
        assert len(user_msgs) == 1  # Only the first user message remains
        assert user_msgs[0]["content"] == "What's on example.com?"
