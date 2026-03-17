"""
Tests for Gemini streaming tool call finish_reason mapping.

Gemini returns finishReason: "STOP" even when tool calls are present.
Per the OpenAI spec, finish_reason must be "tool_calls" when the model
called a tool. The ModelResponseIterator must track tool_calls across
streaming chunks and correctly set finish_reason on the final chunk.

Ref: https://github.com/BerriAI/litellm/issues/21041
"""

from unittest.mock import MagicMock

from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    ModelResponseIterator,
)


def _make_logging_obj(**kwargs):
    """Create a minimal mock logging object for ModelResponseIterator."""
    logging_obj = MagicMock()
    logging_obj.optional_params = kwargs.get("optional_params", {})
    return logging_obj


def test_streaming_tool_call_finish_reason_is_tool_calls():
    """
    When Gemini streams tool calls across two chunks:
      - Chunk 1: has tool call parts, no finishReason
      - Chunk 2: has finishReason="STOP", no content

    The final chunk must have finish_reason="tool_calls" (not "stop").
    """
    logging_obj = _make_logging_obj()
    iterator = ModelResponseIterator(
        streaming_response=iter([]),
        sync_stream=True,
        logging_obj=logging_obj,
    )

    # Chunk 1: tool call with no finishReason
    chunk_with_tool_calls = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_current_weather",
                                "args": {"location": "Boston, MA"},
                            }
                        }
                    ],
                    "role": "model",
                },
                "index": 0,
            }
        ],
    }

    # Chunk 2: finishReason="STOP" with no content
    chunk_with_finish_reason = {
        "candidates": [
            {
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 20,
            "totalTokenCount": 70,
        },
    }

    # Process chunk 1
    response1 = iterator.chunk_parser(chunk_with_tool_calls)
    assert response1 is not None
    assert len(response1.choices) == 1
    assert response1.choices[0].delta.tool_calls is not None
    assert response1.choices[0].finish_reason == "tool_calls"
    assert iterator.has_seen_tool_calls is True

    # Process chunk 2 (final chunk)
    response2 = iterator.chunk_parser(chunk_with_finish_reason)
    assert response2 is not None
    assert len(response2.choices) == 1
    assert response2.choices[0].finish_reason == "tool_calls"


def test_streaming_no_tool_calls_finish_reason_is_stop():
    """
    When Gemini streams a regular text response (no tool calls),
    the final chunk with finishReason="STOP" should map to "stop".
    """
    logging_obj = _make_logging_obj()
    iterator = ModelResponseIterator(
        streaming_response=iter([]),
        sync_stream=True,
        logging_obj=logging_obj,
    )

    # Chunk 1: text content, no finishReason
    chunk_with_text = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello! How can I help?"}],
                    "role": "model",
                },
                "index": 0,
            }
        ],
    }

    # Chunk 2: finishReason="STOP" with no content
    chunk_with_finish_reason = {
        "candidates": [
            {
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 8,
            "totalTokenCount": 18,
        },
    }

    # Process chunk 1
    response1 = iterator.chunk_parser(chunk_with_text)
    assert response1 is not None
    assert len(response1.choices) == 1
    assert iterator.has_seen_tool_calls is False

    # Process chunk 2
    response2 = iterator.chunk_parser(chunk_with_finish_reason)
    assert response2 is not None
    assert len(response2.choices) == 1
    assert response2.choices[0].finish_reason == "stop"


def test_streaming_multiple_tool_calls_finish_reason():
    """
    When Gemini streams multiple tool calls across chunks,
    the final finish_reason must still be "tool_calls".
    """
    logging_obj = _make_logging_obj()
    iterator = ModelResponseIterator(
        streaming_response=iter([]),
        sync_stream=True,
        logging_obj=logging_obj,
    )

    # Chunk 1: first tool call
    chunk_tool_1 = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"location": "NYC"},
                            }
                        },
                        {
                            "functionCall": {
                                "name": "get_time",
                                "args": {"timezone": "EST"},
                            }
                        },
                    ],
                    "role": "model",
                },
                "index": 0,
            }
        ],
    }

    # Chunk 2: finishReason="STOP" with no content
    chunk_finish = {
        "candidates": [
            {
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 30,
            "totalTokenCount": 80,
        },
    }

    response1 = iterator.chunk_parser(chunk_tool_1)
    assert response1 is not None
    assert iterator.has_seen_tool_calls is True

    response2 = iterator.chunk_parser(chunk_finish)
    assert response2 is not None
    assert len(response2.choices) == 1
    assert response2.choices[0].finish_reason == "tool_calls"


def test_streaming_content_filter_finish_reason_preserved():
    """
    When Gemini returns finishReason due to content filtering (not STOP),
    and no tool calls were seen, the content_filter reason should be preserved.
    """
    logging_obj = _make_logging_obj()
    iterator = ModelResponseIterator(
        streaming_response=iter([]),
        sync_stream=True,
        logging_obj=logging_obj,
    )

    # Chunk with finishReason="SAFETY" and no content
    chunk_safety = {
        "candidates": [
            {
                "finishReason": "SAFETY",
                "index": 0,
            }
        ],
    }

    response = iterator.chunk_parser(chunk_safety)
    assert response is not None
    assert len(response.choices) == 1
    assert response.choices[0].finish_reason == "content_filter"
