"""
Unit tests for ZAI/GLM chat transformation.

Tests that list-format content in tool/assistant messages is flattened
to strings before sending to GLM, which requires string-type content.

See: https://github.com/BerriAI/litellm/issues/25868
"""

import pytest

from litellm.llms.zai.chat.transformation import ZAIChatConfig


class TestZAITransformMessages:
    """Test that ZAIChatConfig._transform_messages flattens tool/assistant content."""

    def setup_method(self):
        self.config = ZAIChatConfig()

    def test_tool_message_list_content_flattened(self):
        """Tool message with list content is flattened to string."""
        messages = [
            {"role": "user", "content": "What is 1+1?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "calc", "arguments": '{"x": 1}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "text", "text": "2"}],
            },
        ]

        result = self.config._transform_messages(messages, model="glm-4.6")

        tool_msg = [m for m in result if m.get("role") == "tool"][0]
        assert isinstance(tool_msg["content"], str)
        assert tool_msg["content"] == "2"

    def test_tool_message_string_content_unchanged(self):
        """Tool message with string content passes through."""
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": "result text"},
        ]

        result = self.config._transform_messages(messages, model="glm-4.6")

        assert result[0]["content"] == "result text"

    def test_assistant_message_list_content_flattened(self):
        """Assistant message with list content is flattened."""
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello there"}],
            },
        ]

        result = self.config._transform_messages(messages, model="glm-4.6")

        assert isinstance(result[0]["content"], str)
        assert result[0]["content"] == "Hello there"

    def test_user_message_not_modified(self):
        """User messages are not modified by the ZAI transform.

        User messages may contain image_url content parts that the parent
        class processes — we must not flatten those.
        """
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            },
        ]

        result = self.config._transform_messages(messages, model="glm-4.6")

        assert isinstance(result[0]["content"], list)

    def test_system_message_not_modified(self):
        """System messages are not modified."""
        messages = [
            {"role": "system", "content": "You are helpful."},
        ]

        result = self.config._transform_messages(messages, model="glm-4.6")

        assert result[0]["content"] == "You are helpful."

    def test_tool_message_none_content_unchanged(self):
        """Tool message with None content stays None."""
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": None},
        ]

        result = self.config._transform_messages(messages, model="glm-4.6")

        assert result[0]["content"] is None

    def test_multiple_tool_messages_all_flattened(self):
        """Multiple tool messages with list content are all flattened."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "text", "text": "result 1"}],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": [{"type": "text", "text": "result 2"}],
            },
        ]

        result = self.config._transform_messages(messages, model="glm-4.6")

        assert all(isinstance(m["content"], str) for m in result)
        assert result[0]["content"] == "result 1"
        assert result[1]["content"] == "result 2"

    def test_tool_message_empty_list_becomes_empty_string(self):
        """Tool message with empty list content becomes empty string."""
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": []},
        ]

        result = self.config._transform_messages(messages, model="glm-4.6")

        assert result[0]["content"] == ""

    def test_non_text_content_parts_dropped(self):
        """Non-text content parts (e.g., image_url) in tool messages are dropped."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                    {"type": "text", "text": "caption"},
                ],
            },
        ]

        result = self.config._transform_messages(messages, model="glm-4.6")

        assert isinstance(result[0]["content"], str)
        assert "caption" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_tool_message_list_content_flattened_async(self):
        """Async path: tool message with list content is flattened."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "text", "text": "async result"}],
            },
        ]

        result = await self.config._transform_messages(
            messages, model="glm-4.6", is_async=True
        )

        tool_msg = result[0]
        assert isinstance(tool_msg["content"], str)
        assert tool_msg["content"] == "async result"
