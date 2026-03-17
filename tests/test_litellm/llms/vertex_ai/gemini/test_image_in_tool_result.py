"""
Tests for image handling in tool_result content.

Fixes: https://github.com/BerriAI/litellm/issues/23712

When Claude Code's Read tool returns an image (base64), the Anthropic adapter
converts it to a data URL string. convert_to_gemini_tool_call_result() must
detect this and create inline_data instead of treating it as plain text.
"""

import base64
import pytest

from litellm.litellm_core_utils.prompt_templates.factory import (
    convert_to_gemini_tool_call_result,
)


def _make_tool_message(content, tool_call_id="call_1"):
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }


def _make_last_message(tool_call_id="call_1", name="Read"):
    return {
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ]
    }


class TestImageInToolResult:

    def test_data_url_image_creates_inline_data(self):
        """A data URL image string should produce inline_data, not text."""
        img_data = base64.b64encode(b"fake-png-data").decode()
        data_url = f"data:image/png;base64,{img_data}"

        msg = _make_tool_message(content=data_url)
        last_msg = _make_last_message()

        result = convert_to_gemini_tool_call_result(msg, last_msg)

        # Should return a list: [function_response, inline_data]
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) == 2

        # First part: function_response (with empty content since image was extracted)
        assert "function_response" in result[0]

        # Second part: inline_data with the image
        assert "inline_data" in result[1]
        assert result[1]["inline_data"]["mime_type"] == "image/png"
        assert result[1]["inline_data"]["data"] == img_data

    def test_plain_text_not_affected(self):
        """Regular text content should not be treated as image."""
        msg = _make_tool_message(content="def hello(): pass")
        last_msg = _make_last_message()

        result = convert_to_gemini_tool_call_result(msg, last_msg)

        # Should return single part (not a list)
        assert isinstance(result, dict)
        assert "function_response" in result
        assert result["function_response"]["response"]["content"] == "def hello(): pass"

    def test_multi_item_text_and_image(self):
        """Multi-item content with text + image_url should produce inline_data."""
        img_data = base64.b64encode(b"fake-png").decode()

        msg = _make_tool_message(content=[
            {"type": "text", "text": "Image file: test.png"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_data}"},
            },
        ])
        last_msg = _make_last_message()

        result = convert_to_gemini_tool_call_result(msg, last_msg)

        # Should return list with function_response + inline_data
        assert isinstance(result, list)
        assert len(result) == 2
        assert "function_response" in result[0]
        assert "inline_data" in result[1]

        # Text should be in function_response
        assert "Image file: test.png" in str(result[0]["function_response"]["response"])

    def test_data_url_with_jpeg(self):
        """JPEG data URLs should also be handled."""
        img_data = base64.b64encode(b"fake-jpeg").decode()
        data_url = f"data:image/jpeg;base64,{img_data}"

        msg = _make_tool_message(content=data_url)
        last_msg = _make_last_message()

        result = convert_to_gemini_tool_call_result(msg, last_msg)

        assert isinstance(result, list)
        assert result[1]["inline_data"]["mime_type"] == "image/jpeg"

    def test_non_image_data_url_not_affected(self):
        """Non-image data URLs (e.g. text/plain) should be treated as text."""
        msg = _make_tool_message(content="data:text/plain;base64,aGVsbG8=")
        last_msg = _make_last_message()

        result = convert_to_gemini_tool_call_result(msg, last_msg)

        # Should be a regular text response, not image
        assert isinstance(result, dict)
        assert "function_response" in result
