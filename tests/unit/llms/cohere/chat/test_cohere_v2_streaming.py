"""
Unit tests for the Cohere v2 streaming response iterator.

Wire shapes below are the literal ``data:`` JSON payloads emitted by Cohere's
``/v2/chat`` streaming API. Per Cohere's API reference, every streamed event
carries a top-level ``type`` discriminator and puts its payload under a
top-level ``delta`` key (``index`` is a sibling of ``delta``). For example a
``message-end`` frame is::

    {"type":"message-end",
     "delta":{"finish_reason":"COMPLETE",
              "usage":{"tokens":{"input_tokens":71,"output_tokens":26}}}}

See https://docs.cohere.com/reference/chat-stream (StreamedChatResponseV2).
"""

from litellm.llms.cohere.common_utils import CohereV2ModelResponseIterator


def _iterator() -> CohereV2ModelResponseIterator:
    return CohereV2ModelResponseIterator(streaming_response=iter([]), sync_stream=True)


def test_v2_stream_content_delta():
    """Sanity: text content is still extracted from a content-delta event."""
    chunk = {
        "type": "content-delta",
        "index": 0,
        "delta": {"message": {"content": {"text": "LL"}}},
    }
    parsed = _iterator().chunk_parser(chunk)
    assert parsed["text"] == "LL"
    assert parsed["is_finished"] is False


def test_v2_stream_message_end_sets_finish_reason_and_usage():
    """message-end must surface finish_reason + usage from the stream."""
    chunk = {
        "type": "message-end",
        "delta": {
            "finish_reason": "COMPLETE",
            "usage": {
                "billed_units": {"input_tokens": 5, "output_tokens": 26},
                "tokens": {"input_tokens": 71, "output_tokens": 26},
            },
        },
    }
    parsed = _iterator().chunk_parser(chunk)

    assert parsed["is_finished"] is True
    assert parsed["finish_reason"] == "COMPLETE"
    assert parsed["usage"] is not None
    assert parsed["usage"]["prompt_tokens"] == 71
    assert parsed["usage"]["completion_tokens"] == 26
    assert parsed["usage"]["total_tokens"] == 97


def test_v2_stream_tool_plan_delta():
    """tool-plan-delta must surface the tool plan in provider_specific_fields."""
    chunk = {
        "type": "tool-plan-delta",
        "delta": {"message": {"tool_plan": "I will call the weather tool"}},
    }
    parsed = _iterator().chunk_parser(chunk)
    assert parsed["provider_specific_fields"] is not None
    assert parsed["provider_specific_fields"]["tool_plan"] == "I will call the weather tool"


def test_v2_stream_citation_start():
    """citation-start must surface citations in provider_specific_fields."""
    chunk = {
        "type": "citation-start",
        "index": 0,
        "delta": {
            "message": {
                "citations": {
                    "start": 0,
                    "end": 5,
                    "text": "hello",
                    "sources": [],
                    "type": "TEXT_CONTENT",
                }
            }
        },
    }
    parsed = _iterator().chunk_parser(chunk)
    assert parsed["provider_specific_fields"] is not None
    citations = parsed["provider_specific_fields"]["citations"]
    assert citations[0]["text"] == "hello"
