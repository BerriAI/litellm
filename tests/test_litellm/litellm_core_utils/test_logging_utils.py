"""
Tests for litellm.litellm_core_utils.logging_utils — base64 truncation helpers.
"""

import pytest

from litellm.litellm_core_utils.logging_utils import (
    _format_base64_size,
    _truncate_base64_in_string,
    release_base64_from_request_data_inplace,
    truncate_base64_in_messages,
)

# ---------------------------------------------------------------------------
# _format_base64_size
# ---------------------------------------------------------------------------


class TestFormatBase64Size:
    def test_bytes_range(self):
        assert _format_base64_size(4) == "3B"

    def test_kb_range(self):
        # 2000 base64 chars ~ 1500 bytes ~ 1.5KB
        assert "KB" in _format_base64_size(2000)

    def test_mb_range(self):
        # 2_000_000 base64 chars ~ 1.5MB
        result = _format_base64_size(2_000_000)
        assert "MB" in result


# ---------------------------------------------------------------------------
# _truncate_base64_in_string
# ---------------------------------------------------------------------------


class TestTruncateBase64InString:
    def test_short_data_uri_not_truncated(self):
        uri = "data:image/png;base64,AAAA"
        assert _truncate_base64_in_string(uri) == uri

    def test_long_data_uri_truncated(self):
        payload = "A" * 200
        uri = f"data:application/pdf;base64,{payload}"
        result = _truncate_base64_in_string(uri)
        assert "base64_data truncated" in result
        assert "application/pdf" in result
        assert payload not in result

    def test_multiple_data_uris(self):
        payload = "B" * 200
        text = f"first: data:image/png;base64,{payload} second: data:image/jpeg;base64,{payload}"
        result = _truncate_base64_in_string(text)
        assert result.count("base64_data truncated") == 2

    def test_no_data_uri(self):
        text = "hello world, no base64 here"
        assert _truncate_base64_in_string(text) == text


# ---------------------------------------------------------------------------
# truncate_base64_in_messages
# ---------------------------------------------------------------------------


class TestTruncateBase64InMessages:
    def test_none_input(self):
        assert truncate_base64_in_messages(None) is None

    def test_string_messages(self):
        payload = "C" * 200
        msg = f"Look at data:image/png;base64,{payload}"
        result = truncate_base64_in_messages(msg)
        assert isinstance(result, str)
        assert "base64_data truncated" in result

    def test_openai_vision_format(self):
        """Typical OpenAI multimodal message with image_url containing base64."""
        payload = "D" * 500
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{payload}",
                            "detail": "auto",
                        },
                    },
                ],
            }
        ]
        result = truncate_base64_in_messages(messages)
        # Original must not be mutated
        assert payload in messages[0]["content"][1]["image_url"]["url"]
        # Result should be truncated
        url = result[0]["content"][1]["image_url"]["url"]
        assert "base64_data truncated" in url
        assert payload not in url
        # Non-base64 parts preserved
        assert result[0]["content"][0]["text"] == "What is in this image?"

    def test_multiple_images(self):
        """Two base64 images in one message."""
        payload1 = "E" * 300
        payload2 = "F" * 400
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{payload1}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:application/pdf;base64,{payload2}"},
                    },
                ],
            }
        ]
        result = truncate_base64_in_messages(messages)
        for part in result[0]["content"]:
            assert "base64_data truncated" in part["image_url"]["url"]

    def test_does_not_mutate_original(self):
        payload = "G" * 200
        messages = [{"role": "user", "content": f"data:image/png;base64,{payload}"}]
        truncate_base64_in_messages(messages)
        # Original unchanged
        assert payload in messages[0]["content"]

    def test_dict_messages(self):
        payload = "H" * 200
        messages = {"prompt": f"data:image/png;base64,{payload}"}
        result = truncate_base64_in_messages(messages)
        assert "base64_data truncated" in result["prompt"]

    def test_preserves_short_base64(self):
        """Short base64 under threshold should not be truncated."""
        short = "AAAA"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{short}"},
                    }
                ],
            }
        ]
        result = truncate_base64_in_messages(messages)
        assert result[0]["content"][0]["image_url"]["url"] == f"data:image/png;base64,{short}"


# ---------------------------------------------------------------------------
# release_base64_from_request_data_inplace  (Fix 13)
# ---------------------------------------------------------------------------


class TestReleaseBase64FromRequestDataInplace:
    """Verify in-place base64 truncation for memory cleanup."""

    def test_openai_vision_format_mutated_inplace(self):
        """OpenAI multimodal message: base64 should be truncated in the original dict."""
        payload = "D" * 500
        data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{payload}",
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ],
        }
        release_base64_from_request_data_inplace(data)
        url = data["messages"][0]["content"][1]["image_url"]["url"]
        assert "base64_data truncated" in url
        assert payload not in url
        # Non-base64 parts preserved
        assert data["messages"][0]["content"][0]["text"] == "What is in this image?"
        assert data["model"] == "gpt-4o"

    def test_vertex_gemini_format(self):
        """Vertex AI / Gemini format: inline_data.data should be truncated."""
        payload = "A" * 500
        data = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": payload,
                            }
                        }
                    ],
                }
            ]
        }
        release_base64_from_request_data_inplace(data)
        result_data = data["contents"][0]["parts"][0]["inline_data"]["data"]
        assert "base64_data truncated" in result_data
        assert payload not in result_data

    def test_anthropic_format(self):
        """Anthropic format: source.data should be truncated."""
        payload = "B" * 500
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": payload,
                            },
                        }
                    ],
                }
            ]
        }
        release_base64_from_request_data_inplace(data)
        result_data = data["messages"][0]["content"][0]["source"]["data"]
        assert "base64_data truncated" in result_data
        assert payload not in result_data

    def test_preserves_short_strings(self):
        """Short strings should not be modified."""
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello world"}],
        }
        release_base64_from_request_data_inplace(data)
        assert data["messages"][0]["content"] == "Hello world"
        assert data["model"] == "gpt-4o"

    def test_mutates_same_object(self):
        """The dict should be mutated in-place — same identity."""
        payload = "C" * 500
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": f"data:image/png;base64,{payload}",
                }
            ]
        }
        original_id = id(data)
        original_messages_id = id(data["messages"])
        release_base64_from_request_data_inplace(data)
        assert id(data) == original_id
        assert id(data["messages"]) == original_messages_id
        assert "base64_data truncated" in data["messages"][0]["content"]

    def test_multiple_images(self):
        """Multiple base64 images should all be truncated."""
        p1 = "E" * 300
        p2 = "F" * 400
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{p1}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{p2}"},
                        },
                    ],
                }
            ]
        }
        release_base64_from_request_data_inplace(data)
        for part in data["messages"][0]["content"]:
            assert "base64_data truncated" in part["image_url"]["url"]

    def test_empty_dict(self):
        """Should handle empty dict without error."""
        data = {}
        release_base64_from_request_data_inplace(data)
        assert data == {}

    def test_long_text_not_truncated(self):
        """Long regular text content should NOT be truncated as base64."""
        long_text = "hi" * 5000  # 10000 chars, matches bare base64 regex
        data = {
            "messages": [
                {"role": "user", "content": long_text},
            ]
        }
        release_base64_from_request_data_inplace(data)
        assert data["messages"][0]["content"] == long_text

    def test_data_uri_in_content_string_truncated(self):
        """Data URI embedded in a content string should be truncated."""
        payload = "D" * 500
        text = f"Check this image: data:image/png;base64,{payload}"
        data = {
            "messages": [
                {"role": "user", "content": text},
            ]
        }
        release_base64_from_request_data_inplace(data)
        assert "base64_data truncated" in data["messages"][0]["content"]
        assert payload not in data["messages"][0]["content"]
        assert "Check this image:" in data["messages"][0]["content"]
