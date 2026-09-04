import pytest
from pydantic import ValidationError

from litellm.llms.sap.chat.models import (
    SAPAssistantMessage,
    SAPMessage,
    SAPToolChatMessage,
    SAPUserMessage,
    TextContent,
)


class TestSAPMessage:
    def test_role_system(self):
        msg = SAPMessage(role="system", content="Hi")
        assert msg.role == "system"

    def test_role_developer(self):
        msg = SAPMessage(role="developer", content="Hi")
        assert msg.role == "developer"

    def test_role_defaults_to_system(self):
        msg = SAPMessage(content="Hi")
        assert msg.role == "system"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            SAPMessage(role="user", content="Hi")

    def test_missing_content_rejected(self):
        with pytest.raises(ValidationError):
            SAPMessage(role="system")

    def test_string_content_accepted(self):
        msg = SAPMessage(role="system", content="Hello")
        assert msg.content == "Hello"

    def test_text_content_block_accepted(self):
        msg = SAPMessage(role="system", content={"type": "text", "text": "Hello"})
        assert isinstance(msg.content, TextContent)
        assert msg.content.text == "Hello"

    def test_list_of_text_content_blocks_accepted(self):
        msg = SAPMessage(
            role="system",
            content=[{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
        )
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextContent)

    def test_invalid_content_type_rejected(self):
        with pytest.raises(ValidationError):
            SAPMessage(role="system", content=123)

    def test_cache_control_on_content_block(self):
        msg = SAPMessage(
            role="system",
            content={"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral"}},
        )
        assert isinstance(msg.content, TextContent)
        assert msg.content.cache_control is not None
        assert msg.content.cache_control.type == "ephemeral"


class TestSAPUserMessage:
    def test_role_is_always_user(self):
        msg = SAPUserMessage(content="Hi")
        assert msg.role == "user"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            SAPUserMessage(role="system", content="Hi")

    def test_missing_content_rejected(self):
        with pytest.raises(ValidationError):
            SAPUserMessage()

    def test_string_content_accepted(self):
        msg = SAPUserMessage(content="Hello")
        assert msg.content == "Hello"

    def test_text_content_block_accepted(self):
        msg = SAPUserMessage(content={"type": "text", "text": "Hello"})
        assert isinstance(msg.content, TextContent)

    def test_image_content_accepted(self):
        from litellm.llms.sap.chat.models import ImageContent

        msg = SAPUserMessage(
            content={"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
        )
        assert isinstance(msg.content, ImageContent)

    def test_mixed_list_of_text_and_image_accepted(self):
        msg = SAPUserMessage(
            content=[
                {"type": "text", "text": "Look at this:"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ]
        )
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_invalid_content_type_rejected(self):
        with pytest.raises(ValidationError):
            SAPUserMessage(content=123)

    def test_cache_control_on_content_block(self):
        msg = SAPUserMessage(
            content={"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral"}},
        )
        assert isinstance(msg.content, TextContent)
        assert msg.content.cache_control is not None
        assert msg.content.cache_control.type == "ephemeral"


class TestSAPAssistantMessage:
    def test_role_is_always_assistant(self):
        msg = SAPAssistantMessage(content="Hi")
        assert msg.role == "assistant"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            SAPAssistantMessage(role="user", content="Hi")

    def test_refusal_defaults_to_empty_string(self):
        msg = SAPAssistantMessage()
        assert msg.refusal == ""

    def test_refusal_accepted(self):
        msg = SAPAssistantMessage(refusal="I cannot help with that.")
        assert msg.refusal == "I cannot help with that."

    def test_tool_calls_default_to_empty_list(self):
        msg = SAPAssistantMessage()
        assert msg.tool_calls == []

    def test_string_content_accepted(self):
        msg = SAPAssistantMessage(content="Hello")
        assert msg.content == "Hello"

    def test_default_empty_string(self):
        msg = SAPAssistantMessage()
        assert msg.content == ""

    def test_text_content_block_accepted(self):
        msg = SAPAssistantMessage(content={"type": "text", "text": "Hello"})
        assert isinstance(msg.content, TextContent)

    def test_list_of_text_blocks_accepted(self):
        msg = SAPAssistantMessage(
            content=[{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]
        )
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_invalid_content_type_rejected(self):
        with pytest.raises(ValidationError):
            SAPAssistantMessage(content={"type": "image_url", "image_url": {"url": "x"}})

    def test_cache_control_on_content_block(self):
        msg = SAPAssistantMessage(
            content={"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral"}},
        )
        assert isinstance(msg.content, TextContent)
        assert msg.content.cache_control is not None
        assert msg.content.cache_control.type == "ephemeral"


class TestSAPToolChatMessage:
    def test_role_is_always_tool(self):
        msg = SAPToolChatMessage(tool_call_id="call_1", content="ok")
        assert msg.role == "tool"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            SAPToolChatMessage(role="user", tool_call_id="call_1", content="ok")

    def test_tool_call_id_accepted(self):
        msg = SAPToolChatMessage(tool_call_id="call_abc123", content="result")
        assert msg.tool_call_id == "call_abc123"

    def test_missing_tool_call_id_rejected(self):
        with pytest.raises(ValidationError):
            SAPToolChatMessage(content="result")

    def test_string_content_accepted(self):
        msg = SAPToolChatMessage(tool_call_id="call_1", content="result")
        assert msg.content == "result"

    def test_text_content_block_accepted(self):
        msg = SAPToolChatMessage(
            tool_call_id="call_1", content={"type": "text", "text": "result"}
        )
        assert isinstance(msg.content, TextContent)

    def test_list_of_text_blocks_accepted(self):
        msg = SAPToolChatMessage(
            tool_call_id="call_1",
            content=[{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
        )
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_missing_content_rejected(self):
        with pytest.raises(ValidationError):
            SAPToolChatMessage(tool_call_id="call_1")

    def test_invalid_content_type_rejected(self):
        with pytest.raises(ValidationError):
            SAPToolChatMessage(tool_call_id="call_1", content=99)

    def test_cache_control_on_content_block(self):
        msg = SAPToolChatMessage(
            tool_call_id="call_1",
            content={"type": "text", "text": "result", "cache_control": {"type": "ephemeral"}},
        )
        assert isinstance(msg.content, TextContent)
        assert msg.content.cache_control is not None
        assert msg.content.cache_control.type == "ephemeral"
