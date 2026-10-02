"""
Tests for DeepSeek vision-content forwarding in role=tool messages.

Bug #44211: ``_is_vision_forwardable_content`` rejected every non-user role,
so a ``role=tool`` message carrying image_url blocks was silently collapsed
to text before the request left LiteLLM — while the DeepSeek API itself
accepts and reads tool-result images.
"""

import pytest

from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


TOOL_MESSAGE_WITH_IMAGE = {
    "role": "tool",
    "tool_call_id": "abc",
    "content": [
        {"type": "text", "text": "screenshot:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
    ],
}


class TestVisionForwardableContent:
    def test_tool_message_with_image_is_forwardable(self):
        config = DeepSeekChatConfig()
        assert config._is_vision_forwardable_content(
            message=TOOL_MESSAGE_WITH_IMAGE,
            content=TOOL_MESSAGE_WITH_IMAGE["content"],
        )

    def test_user_message_with_image_stays_forwardable(self):
        config = DeepSeekChatConfig()
        message = {"role": "user", "content": TOOL_MESSAGE_WITH_IMAGE["content"]}
        assert config._is_vision_forwardable_content(message=message, content=message["content"])

    def test_assistant_message_stays_collapsed(self):
        config = DeepSeekChatConfig()
        message = {"role": "assistant", "content": TOOL_MESSAGE_WITH_IMAGE["content"]}
        assert not config._is_vision_forwardable_content(message=message, content=message["content"])

    def test_image_missing_payload_falls_back(self):
        config = DeepSeekChatConfig()
        message = {
            "role": "tool",
            "content": [
                {"type": "text", "text": "screenshot:"},
                {"type": "image_url", "image_url": {"url": ""}},
            ],
        }
        assert not config._is_vision_forwardable_content(message=message, content=message["content"])


class TestForwardOrCollapseContent:
    def test_tool_message_image_content_is_not_collapsed(self):
        config = DeepSeekChatConfig()
        out = config._forward_or_collapse_content(message=TOOL_MESSAGE_WITH_IMAGE, forward_images=True)
        assert isinstance(out.get("content"), list)
        blocks = out["content"]
        assert any(isinstance(block, dict) and block.get("type") == "image_url" for block in blocks)

    def test_tool_message_text_only_still_collapsed(self):
        config = DeepSeekChatConfig()
        message = {"role": "tool", "content": [{"type": "text", "text": "plain"}]}
        out = config._forward_or_collapse_content(message=message, forward_images=True)
        assert out.get("content") == "plain"
