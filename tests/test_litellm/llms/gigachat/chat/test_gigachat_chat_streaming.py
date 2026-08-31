"""
Tests for litellm.llms.gigachat.chat.streaming
"""

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

    def test_cache_hit_usage_not_inflated(self):
        """precached_prompt_tokens is a subset of prompt_tokens; totals must not be inflated on cache hits."""
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
        assert parsed["usage"]["prompt_tokens"] == 25
        assert parsed["usage"]["total_tokens"] == 32
        assert parsed["usage"]["prompt_tokens_details"]["cached_tokens"] == 20
