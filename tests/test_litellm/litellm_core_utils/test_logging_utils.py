"""
Tests for litellm.litellm_core_utils.logging_utils — base64 truncation helpers.
"""

import threading

import pytest

from litellm.litellm_core_utils import logging_utils
from litellm.litellm_core_utils.logging_utils import (
    _format_base64_size,
    _truncate_base64_in_string,
    truncate_base64_in_messages,
    truncate_base64_in_messages_async,
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
        assert (
            result[0]["content"][0]["image_url"]["url"]
            == f"data:image/png;base64,{short}"
        )


# ---------------------------------------------------------------------------
# truncate_base64_in_messages_async
# ---------------------------------------------------------------------------


def _image_messages(payload: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}},
            ],
        }
    ]


@pytest.fixture
def scan_threads(monkeypatch):
    """Record the thread that runs every base64 regex scan."""
    threads: list[int] = []
    original = logging_utils._truncate_base64_in_string

    def recording_scan(value: str) -> str:
        threads.append(threading.get_ident())
        return original(value)

    monkeypatch.setattr(logging_utils, "_truncate_base64_in_string", recording_scan)
    return threads


class TestTruncateBase64InMessagesAsync:
    @pytest.mark.asyncio
    async def test_large_payload_is_scanned_off_the_event_loop(self, monkeypatch, scan_threads):
        monkeypatch.setattr(logging_utils, "BASE64_TRUNCATION_OFFLOAD_THRESHOLD_CHARS", 1_000)
        payload = "I" * 20_000
        messages = _image_messages(payload)

        result = await truncate_base64_in_messages_async(messages)
        offload_threads = tuple(scan_threads)

        assert result == truncate_base64_in_messages(messages)
        assert payload not in result[0]["content"][1]["image_url"]["url"]
        assert payload in messages[0]["content"][1]["image_url"]["url"]
        assert offload_threads
        assert threading.get_ident() not in offload_threads

    @pytest.mark.asyncio
    async def test_small_payload_stays_on_the_calling_thread(self, monkeypatch, scan_threads):
        monkeypatch.setattr(logging_utils, "BASE64_TRUNCATION_OFFLOAD_THRESHOLD_CHARS", 1_000)
        messages = _image_messages("J" * 200)

        result = await truncate_base64_in_messages_async(messages)

        assert result == truncate_base64_in_messages(messages)
        assert scan_threads
        assert set(scan_threads) == {threading.get_ident()}

    @pytest.mark.asyncio
    async def test_none_and_disabled_truncation_short_circuit(self, monkeypatch, scan_threads):
        assert await truncate_base64_in_messages_async(None) is None
        monkeypatch.setattr(logging_utils, "MAX_BASE64_LENGTH_FOR_LOGGING", 0)
        messages = _image_messages("K" * 20_000)
        assert await truncate_base64_in_messages_async(messages) is messages
        assert scan_threads == []
