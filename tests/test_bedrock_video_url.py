"""
Tests for issue #40681: Bedrock Converse silently drops video_url content.

When sending video_url content blocks to Bedrock Converse models, they must
be converted to BedrockVideoBlock instead of being silently dropped.
"""
import base64
import pytest
from unittest.mock import patch, MagicMock

from litellm.litellm_core_utils.prompt_templates.factory import (
    _bedrock_converse_messages_pt,
    _parse_bedrock_tool_result_content_list,
)


def _make_video_base64():
    """Create a minimal valid MP4 base64 data URI for testing."""
    # Minimal valid MP4 header (ftyp box)
    mp4_bytes = (
        b"\x00\x00\x00\x1c\x66\x74\x79\x70\x6d\x70\x34\x32"  # ftyp box
        b"\x00\x00\x00\x00\x6d\x70\x34\x32\x6d\x70\x34\x31"  # brands
        b"\x69\x73\x6f\x6d\x00\x00\x00\x08\x66\x72\x65\x65"  # free box
    )
    b64 = base64.b64encode(mp4_bytes).decode()
    return f"data:video/mp4;base64,{b64}"


class TestBedrockConverseVideoUrl:
    """Test that video_url content blocks are converted to BedrockVideoBlock."""

    def test_video_url_in_user_message_sync(self):
        """video_url in user message must produce a BedrockVideoBlock."""
        video_data_uri = _make_video_base64()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video."},
                    {
                        "type": "video_url",
                        "video_url": {"url": video_data_uri},
                    },
                ],
            }
        ]

        result = _bedrock_converse_messages_pt(
            messages=messages,
            model="amazon.nova-pro-v1:0",
            llm_provider="bedrock",
        )

        assert len(result) == 1
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        # Should have text block + video block
        assert len(content) == 2
        assert content[0].get("text") == "Describe this video."
        assert "video" in content[1], (
            f"Expected 'video' key in content block, got: {list(content[1].keys())}"
        )
        video_block = content[1]["video"]
        assert "source" in video_block
        assert video_block["format"] == "mp4"

    def test_video_url_string_format(self):
        """video_url as a plain string (not dict) must also work."""
        video_data_uri = _make_video_base64()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": video_data_uri,
                    },
                ],
            }
        ]

        result = _bedrock_converse_messages_pt(
            messages=messages,
            model="amazon.nova-pro-v1:0",
            llm_provider="bedrock",
        )

        assert len(result) == 1
        content = result[0]["content"]
        assert len(content) == 1
        assert "video" in content[0], (
            f"Expected 'video' key in content block, got: {list(content[0].keys())}"
        )

    def test_video_url_with_explicit_format(self):
        """video_url with explicit format override must use that format."""
        video_data_uri = _make_video_base64()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": video_data_uri, "format": "video/mp4"},
                    },
                ],
            }
        ]

        result = _bedrock_converse_messages_pt(
            messages=messages,
            model="amazon.nova-pro-v1:0",
            llm_provider="bedrock",
        )

        content = result[0]["content"]
        assert "video" in content[0]
        assert content[0]["video"]["format"] == "mp4"

    def test_video_url_in_tool_result(self):
        """video_url in tool result content must produce a video block."""
        video_data_uri = _make_video_base64()
        content_list = [
            {"type": "text", "text": "Here is the video"},
            {
                "type": "video_url",
                "video_url": {"url": video_data_uri},
            },
        ]

        result = _parse_bedrock_tool_result_content_list(content_list)

        assert len(result) == 2
        assert result[0].get("text") == "Here is the video"
        assert "video" in result[1], (
            f"Expected 'video' key in tool result block, got: {list(result[1].keys())}"
        )

    def test_video_url_mixed_with_image_url(self):
        """Both video_url and image_url must coexist in a single message."""
        video_data_uri = _make_video_base64()
        # Create a minimal PNG
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        img_b64 = base64.b64encode(png_bytes).decode()
        img_data_uri = f"data:image/png;base64,{img_b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these"},
                    {
                        "type": "image_url",
                        "image_url": {"url": img_data_uri},
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": video_data_uri},
                    },
                ],
            }
        ]

        result = _bedrock_converse_messages_pt(
            messages=messages,
            model="amazon.nova-pro-v1:0",
            llm_provider="bedrock",
        )

        content = result[0]["content"]
        # Should have: text, image, video
        assert len(content) == 3
        assert content[0].get("text") == "Compare these"
        assert "image" in content[1], (
            f"Expected 'image' key, got: {list(content[1].keys())}"
        )
        assert "video" in content[2], (
            f"Expected 'video' key, got: {list(content[2].keys())}"
        )
