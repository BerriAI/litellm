"""
Tests for litellm.llms.gigachat.chat.streaming
"""

import json
from collections.abc import AsyncIterator
from typing import Final

from litellm.llms.gigachat.chat.streaming import GigaChatModelResponseIterator


def _parse(chunk: dict) -> dict:
    iterator = GigaChatModelResponseIterator(streaming_response=None, sync_stream=True)
    return dict(iterator.chunk_parser(chunk=chunk))


class TestChunkParserUsage:
    def test_usage_on_stop_chunk(self):
        parsed = _parse(
            {
                "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 25, "completion_tokens": 7, "total_tokens": 32},
            }
        )

        assert parsed["finish_reason"] == "stop"
        assert parsed["usage"] is not None
        assert parsed["usage"]["prompt_tokens"] == 25
        assert parsed["usage"]["completion_tokens"] == 7
        assert parsed["usage"]["total_tokens"] == 32

    def test_usage_on_function_call_chunk(self):
        """Regression: a final chunk ending in function_call still carries usage; it must not be dropped."""
        parsed = _parse(
            {
                "choices": [
                    {
                        "delta": {"function_call": {"name": "get_weather", "arguments": {"city": "Moscow"}}},
                        "index": 0,
                        "finish_reason": "function_call",
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
            }
        )

        assert parsed["finish_reason"] == "tool_calls"
        assert parsed["tool_use"] is not None
        assert parsed["usage"] is not None
        assert parsed["usage"]["prompt_tokens"] == 40
        assert parsed["usage"]["completion_tokens"] == 12
        assert parsed["usage"]["total_tokens"] == 52

    def test_usage_on_length_chunk(self):
        parsed = _parse(
            {
                "choices": [{"delta": {"content": "truncated"}, "index": 0, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 128, "total_tokens": 138},
            }
        )

        assert parsed["usage"] is not None
        assert parsed["usage"]["total_tokens"] == 138

    def test_no_usage_on_interim_chunk(self):
        parsed = _parse({"choices": [{"delta": {"content": "hello"}, "index": 0, "finish_reason": None}]})

        assert parsed["text"] == "hello"
        assert parsed["is_finished"] is False
        assert parsed["usage"] is None

    def test_cache_hit_usage_folds_cached_tokens_back_in(self):
        """GigaChat reports prompt_tokens and total_tokens after subtracting cached tokens
        (docs example: prompt_tokens=1, precached_prompt_tokens=37, total_tokens=5), so the
        OpenAI-convention usage must add them back and surface them as cached_tokens."""
        parsed = _parse(
            {
                "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 25,
                    "completion_tokens": 7,
                    "total_tokens": 32,
                    "precached_prompt_tokens": 20,
                },
            }
        )

        assert parsed["usage"] is not None
        assert parsed["usage"]["prompt_tokens"] == 45
        assert parsed["usage"]["total_tokens"] == 52
        assert parsed["usage"]["prompt_tokens_details"]["cached_tokens"] == 20


def _content_chunk(text: str) -> dict[str, object]:
    return {"choices": [{"delta": {"content": text}, "index": 0, "finish_reason": None}]}


STREAM_ITEMS: Final = (
    f"data: {json.dumps(_content_chunk('Hel'))}",
    _content_chunk("lo"),
    "data: {not json",
    "data: [DONE]",
    f"data: {json.dumps(_content_chunk('after done'))}",
)


class TestStreamIteration:
    def test_sync_stream_reads_sse_lines_and_parsed_chunks_until_done(self):
        iterator = GigaChatModelResponseIterator(streaming_response=iter(STREAM_ITEMS), sync_stream=True)

        assert [chunk["text"] for chunk in iterator] == ["Hel", "lo", ""]

    async def test_async_stream_reads_sse_lines_and_parsed_chunks_until_done(self):
        async def stream() -> AsyncIterator[object]:
            for item in STREAM_ITEMS:
                yield item

        iterator = GigaChatModelResponseIterator(streaming_response=stream(), sync_stream=False)

        assert [chunk["text"] async for chunk in iterator] == ["Hel", "lo", ""]
