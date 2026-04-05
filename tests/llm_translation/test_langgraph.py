"""
Tests for LangGraph provider integration.

These tests require a LangGraph server running locally on port 2024.
To start a LangGraph server, follow the LangGraph documentation.

Example test server curl commands:
Streaming:
  curl -s --request POST \
    --url "http://localhost:2024/runs/stream" \
    --header 'Content-Type: application/json' \
    --data '{"assistant_id": "agent", "input": {"messages": [{"role": "human", "content": "What is 25 * 4?"}]}, "stream_mode": "messages-tuple"}'

Non-streaming:
  curl -s --request POST \
    --url "http://localhost:2024/runs/wait" \
    --header 'Content-Type: application/json' \
    --data '{"assistant_id": "agent", "input": {"messages": [{"role": "human", "content": "What is 25 * 4?"}]}}'
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import pytest

import litellm


@pytest.mark.asyncio
async def test_langgraph_acompletion_non_streaming():
    """
    Test non-streaming acompletion call to LangGraph server.
    Uses the /runs/wait endpoint for synchronous response.
    """
    api_base = os.environ.get("LANGGRAPH_API_BASE", "http://localhost:2024")

    try:
        response = await litellm.acompletion(
            model="langgraph/agent",
            messages=[{"role": "user", "content": "What is 25 * 4?"}],
            api_base=api_base,
            stream=False,
        )

        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message is not None
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    except Exception as e:
        pytest.skip(f"LangGraph server not available: {e}")


@pytest.mark.asyncio
async def test_langgraph_acompletion_streaming():
    """
    Test streaming acompletion call to LangGraph server.
    Uses the /runs/stream endpoint with stream_mode="messages-tuple".
    """
    api_base = os.environ.get("LANGGRAPH_API_BASE", "http://localhost:2024")

    try:
        response = await litellm.acompletion(
            model="langgraph/agent",
            messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
            api_base=api_base,
            stream=True,
        )

        full_content = ""
        chunk_count = 0

        async for chunk in response:
            chunk_count += 1
            if (
                chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                full_content += chunk.choices[0].delta.content

        assert chunk_count > 0, "Should receive at least one chunk"

    except Exception as e:
        pytest.skip(f"LangGraph server not available: {e}")


def test_langgraph_config_get_complete_url():
    """
    Test that LangGraphConfig correctly generates URLs for streaming and non-streaming.
    """
    from litellm.llms.langgraph.chat.transformation import LangGraphConfig

    config = LangGraphConfig()

    non_streaming_url = config.get_complete_url(
        api_base="http://localhost:2024",
        api_key=None,
        model="agent",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert non_streaming_url == "http://localhost:2024/runs/wait"

    streaming_url = config.get_complete_url(
        api_base="http://localhost:2024",
        api_key=None,
        model="agent",
        optional_params={},
        litellm_params={},
        stream=True,
    )
    assert streaming_url == "http://localhost:2024/runs/stream"


def test_langgraph_config_transform_request():
    """
    Test that LangGraphConfig correctly transforms requests.
    """
    from litellm.llms.langgraph.chat.transformation import LangGraphConfig

    config = LangGraphConfig()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]

    request = config.transform_request(
        model="langgraph/agent",
        messages=messages,
        optional_params={},
        litellm_params={"stream": False},
        headers={},
    )

    assert request["assistant_id"] == "agent"
    assert "input" in request
    assert "messages" in request["input"]
    assert len(request["input"]["messages"]) == 2
    assert request["input"]["messages"][0]["role"] == "system"
    assert request["input"]["messages"][1]["role"] == "human"

    streaming_request = config.transform_request(
        model="langgraph/agent",
        messages=messages,
        optional_params={},
        litellm_params={"stream": True},
        headers={},
    )

    assert streaming_request["stream_mode"] == "messages-tuple"


def test_langgraph_provider_detection():
    """
    Test that the langgraph provider is correctly detected from model name.
    """
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider(
        model="langgraph/agent",
        api_base="http://localhost:2024",
    )

    assert provider == "langgraph"
    assert model == "agent"


# ---------------------------------------------------------------------------
# Unit tests for LangGraphSSEStreamIterator — issue #24093
# ---------------------------------------------------------------------------


def _make_iterator(model: str = "agent"):
    from litellm.llms.langgraph.chat.sse_iterator import LangGraphSSEStreamIterator

    # Build a bare instance without needing a real httpx.Response.
    iterator = LangGraphSSEStreamIterator.__new__(LangGraphSSEStreamIterator)
    iterator.model = model
    iterator.finished = False
    iterator.line_iterator = None
    iterator.async_line_iterator = None
    iterator._current_event = None
    return iterator


_AI_CHUNK = {
    "content": "Hello, world!",
    "additional_kwargs": {},
    "response_metadata": {"finish_reason": "stop"},
    "type": "AIMessageChunk",
    "name": None,
    "id": "lc_run--test",
    "tool_calls": [],
    "invalid_tool_calls": [],
    "usage_metadata": None,
    "tool_call_chunks": [],
    "chunk_position": None,
}

_METADATA = {
    "created_by": "system",
    "run_id": "019d04b0-f4bf-7842-a109-bccee3c1bf33",
    "thread_id": "d475429f-e595-48dc-a7fa-066afaabf43c",
}


def test_sse_event_line_sets_current_event():
    """``event:`` lines are tracked in _current_event."""
    it = _make_iterator()
    result = it._parse_sse_line("event: messages")
    assert result is None  # event lines don't produce output
    assert it._current_event == "messages"


def test_sse_empty_line_resets_current_event():
    """An empty line (SSE event boundary) resets _current_event."""
    it = _make_iterator()
    it._current_event = "messages"
    it._parse_sse_line("")
    assert it._current_event is None


def test_sse_standard_messages_event_produces_chunk():
    """
    The standard LangGraph SSE frame::

        event: messages
        data: [<AIMessageChunk>, <metadata>]

    must produce a content chunk.

    Regression (issue #24093): the data array is NOT ``[event_type, payload]``
    but ``[msg_dict, meta_dict]``.  The old code treated ``data[0]`` as the
    event type and therefore never matched ``"messages"``, silently dropping
    every streaming frame.
    """
    import json

    it = _make_iterator()
    it._parse_sse_line("event: messages")
    assert it._current_event == "messages"

    data_line = "data: " + json.dumps([_AI_CHUNK, _METADATA])
    chunk = it._parse_sse_line(data_line)

    assert chunk is not None, (
        "Expected a content chunk from a standard SSE messages event, got None. "
        "This is the regression from issue #24093."
    )
    assert len(chunk.choices) == 1
    assert chunk.choices[0].delta.content == "Hello, world!"


def test_sse_data_without_event_header_is_safe():
    """
    A ``data:`` line whose payload is [dict, dict] but has no preceding
    ``event:`` line must not crash and must not accidentally match the legacy
    tuple path (which requires data[0] to be a string).
    """
    import json

    it = _make_iterator()
    # No event: header — _current_event is None
    data_line = "data: " + json.dumps([_AI_CHUNK, _METADATA])
    chunk = it._parse_sse_line(data_line)
    assert chunk is None


def test_sse_metadata_event_via_header():
    """
    Standard SSE ``event: metadata`` frame with a dict payload containing
    ``run_id`` should set finished=True and return a final chunk.
    """
    import json

    it = _make_iterator()
    it._parse_sse_line("event: metadata")
    chunk = it._parse_sse_line("data: " + json.dumps(_METADATA))

    assert chunk is not None
    assert chunk.choices[0].finish_reason == "stop"
    assert it.finished is True


def test_sse_multiple_messages_frames_each_produce_chunk():
    """Consecutive SSE events should each produce an independent chunk."""
    import json

    it = _make_iterator()
    collected = []

    for text in ("The", " answer", " is 42."):
        msg = dict(_AI_CHUNK)
        msg["content"] = text
        it._parse_sse_line("event: messages")
        chunk = it._parse_sse_line("data: " + json.dumps([msg, _METADATA]))
        it._parse_sse_line("")  # SSE event boundary
        if chunk is not None:
            collected.append(chunk)

    assert len(collected) == 3
    assert [c.choices[0].delta.content for c in collected] == ["The", " answer", " is 42."]


def test_sse_empty_content_ai_chunk_returns_none():
    """AIMessageChunk with empty content should not produce a chunk."""
    import json

    it = _make_iterator()
    msg = dict(_AI_CHUNK)
    msg["content"] = ""
    it._parse_sse_line("event: messages")
    chunk = it._parse_sse_line("data: " + json.dumps([msg, _METADATA]))
    assert chunk is None


def test_sse_legacy_tuple_format_still_works():
    """
    Backward-compat: some clients may use the legacy ``["messages", payload]``
    data format without a preceding ``event:`` line.  The fallback path must
    continue to handle this.
    """
    import json

    it = _make_iterator()
    # No event: header — _current_event is None; data[0] is a string → legacy path
    legacy_payload = [_AI_CHUNK]
    data_line = "data: " + json.dumps(["messages", legacy_payload])
    chunk = it._parse_sse_line(data_line)

    assert chunk is not None
    assert chunk.choices[0].delta.content == "Hello, world!"


def test_sse_full_stream_simulation():
    """
    Simulate a realistic multi-frame LangGraph SSE stream and verify that all
    content frames are collected and the final metadata chunk has finish_reason.
    """
    import json

    it = _make_iterator()
    collected = []

    # Three content frames
    for text in ("The answer", " is", " 42."):
        msg = dict(_AI_CHUNK)
        msg["content"] = text
        for line in [
            "event: messages",
            "data: " + json.dumps([msg, _METADATA]),
            "",
        ]:
            chunk = it._parse_sse_line(line)
            if chunk is not None:
                collected.append(chunk)

    # Final metadata frame
    for line in ["event: metadata", "data: " + json.dumps(_METADATA), ""]:
        chunk = it._parse_sse_line(line)
        if chunk is not None:
            collected.append(chunk)

    content_chunks = [c for c in collected if c.choices[0].finish_reason is None]
    final_chunks = [c for c in collected if c.choices[0].finish_reason == "stop"]

    assert len(content_chunks) == 3
    assert "".join(c.choices[0].delta.content for c in content_chunks) == "The answer is 42."
    assert len(final_chunks) == 1

