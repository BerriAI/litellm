"""
XCT fork: A2A agent backends (xct-agent-gateway) are OpenAI-compatible and
return a chat.completion(.chunk) envelope rather than A2A JSON-RPC. Without a
fallback, extract_text_from_a2a_response() returns "" and the agent appears to
reply with empty content. These tests lock in the OpenAI-envelope fallback.
"""

from litellm.llms.a2a.common_utils import (
    extract_text_from_a2a_response,
    extract_text_from_openai_envelope,
)
from litellm.llms.a2a.chat.streaming_iterator import A2AModelResponseIterator


class TestOpenAIEnvelopeFallback:
    def test_non_stream_openai_envelope(self):
        resp = {
            "id": "abc",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hello from agent"},
                }
            ],
        }
        assert extract_text_from_a2a_response(resp) == "Hello from agent"

    def test_stream_chunk_delta(self):
        chunk = {
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "partial"}}],
        }
        assert extract_text_from_a2a_response(chunk) == "partial"

    def test_native_a2a_result_still_works(self):
        # Regression: real A2A responses must keep working (fallback must not shadow).
        a2a = {
            "result": {
                "kind": "message",
                "parts": [{"kind": "text", "text": "a2a text"}],
            }
        }
        assert extract_text_from_a2a_response(a2a) == "a2a text"

    def test_empty_when_neither_shape(self):
        assert extract_text_from_a2a_response({"foo": "bar"}) == ""

    def test_envelope_helper_handles_missing_content(self):
        assert extract_text_from_openai_envelope({"choices": [{"message": {}}]}) == ""
        assert extract_text_from_openai_envelope({"choices": []}) == ""
        assert extract_text_from_openai_envelope({}) == ""


class TestStreamingFinishReason:
    def _iter(self):
        return A2AModelResponseIterator(streaming_response=iter([]), sync_stream=True)

    def test_openai_finish_reason_terminates_stream(self):
        it = self._iter()
        terminal = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        parsed = it.chunk_parser(terminal)
        assert parsed["is_finished"] is True
        assert parsed["finish_reason"] == "stop"

    def test_openai_content_chunk_not_finished(self):
        it = self._iter()
        chunk = {"choices": [{"delta": {"content": "hi"}}]}
        parsed = it.chunk_parser(chunk)
        assert parsed["text"] == "hi"
        assert parsed["is_finished"] is False
