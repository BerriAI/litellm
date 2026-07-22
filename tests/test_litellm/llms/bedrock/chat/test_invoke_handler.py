import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.bedrock.chat.invoke_handler import (
    AWSEventStreamDecoder,
    BedrockError,
    BedrockLLM,
    make_call,
    make_sync_call,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler


def _mock_event(body: dict) -> MagicMock:
    """Build a botocore-style event whose 200 body is the given ConverseStream JSON."""
    event = MagicMock()
    event.to_response_dict.return_value = {
        "status_code": 200,
        "headers": {},
        "body": json.dumps(body).encode(),
    }
    return event


def _decode_events_sync(decoder: AWSEventStreamDecoder, bodies: list[dict]):
    mock_buffer = MagicMock()
    mock_buffer.__iter__.return_value = [_mock_event(b) for b in bodies]
    with patch("botocore.eventstream.EventStreamBuffer", return_value=mock_buffer):
        return list(decoder.iter_bytes(iter([b"raw"])))


async def _decode_events_async(decoder: AWSEventStreamDecoder, bodies: list[dict]):
    mock_buffer = MagicMock()
    mock_buffer.__iter__.return_value = [_mock_event(b) for b in bodies]

    async def _aiter():
        yield b"raw"

    with patch("botocore.eventstream.EventStreamBuffer", return_value=mock_buffer):
        return [chunk async for chunk in decoder.aiter_bytes(_aiter())]


_TRUNCATED_TOOL_CALL_STREAM = [
    {"messageStart": {"role": "assistant"}},
    {"start": {"toolUse": {"toolUseId": "tooluse_abc", "name": "shell"}}, "contentBlockIndex": 0},
    {"delta": {"toolUse": {"input": '{"command": ["cat",".glia/project.md"]'}}, "contentBlockIndex": 0},
]

_COMPLETE_TOOL_CALL_STREAM = _TRUNCATED_TOOL_CALL_STREAM + [
    {"delta": {"toolUse": {"input": "}"}}, "contentBlockIndex": 0},
    {"contentBlockIndex": 0},
    {"stopReason": "tool_use"},
    {"usage": {"inputTokens": 10641, "outputTokens": 48, "totalTokens": 10689}, "metrics": {"latencyMs": 100}},
]


def test_converse_stream_without_message_stop_raises_sync():
    """A ConverseStream that ends mid-tool-call without a terminal messageStop
    (stopReason) event must surface as an error, not be silently completed with a
    fabricated finish_reason and a synthesized usage chunk. Regression for #32686."""
    decoder = AWSEventStreamDecoder(model="anthropic.claude-opus-4-1-20250805-v1:0")
    with pytest.raises(BedrockError) as exc_info:
        _decode_events_sync(decoder, _TRUNCATED_TOOL_CALL_STREAM)
    assert exc_info.value.status_code == 500
    assert "messageStop" in exc_info.value.message


@pytest.mark.asyncio
async def test_converse_stream_without_message_stop_raises_async():
    decoder = AWSEventStreamDecoder(model="anthropic.claude-opus-4-1-20250805-v1:0")
    with pytest.raises(BedrockError) as exc_info:
        await _decode_events_async(decoder, _TRUNCATED_TOOL_CALL_STREAM)
    assert exc_info.value.status_code == 500
    assert "messageStop" in exc_info.value.message


def test_converse_stream_with_message_stop_does_not_raise_sync():
    """A well-formed ConverseStream ending in messageStop + metadata must not raise."""
    decoder = AWSEventStreamDecoder(model="anthropic.claude-opus-4-1-20250805-v1:0")
    chunks = _decode_events_sync(decoder, _COMPLETE_TOOL_CALL_STREAM)
    finish_reasons = [
        c.choices[0].finish_reason for c in chunks if hasattr(c, "choices") and c.choices and c.choices[0].finish_reason
    ]
    assert "tool_calls" in finish_reasons


@pytest.mark.asyncio
async def test_converse_stream_with_message_stop_does_not_raise_async():
    decoder = AWSEventStreamDecoder(model="anthropic.claude-opus-4-1-20250805-v1:0")
    chunks = await _decode_events_async(decoder, _COMPLETE_TOOL_CALL_STREAM)
    assert any(hasattr(c, "choices") and c.choices and c.choices[0].finish_reason == "tool_calls" for c in chunks)


def test_non_converse_invoke_stream_end_without_stop_reason_does_not_raise():
    """The messageStop guard is Converse-only; a non-Converse invoke text stream
    (e.g. cohere) that never routes through converse_chunk_parser must not be
    affected by the incomplete-stream check."""
    decoder = AWSEventStreamDecoder(model="cohere.command-text-v14")
    chunks = _decode_events_sync(decoder, [{"text": "hello"}, {"text": " world"}])
    assert [c["text"] for c in chunks] == ["hello", " world"]


def test_transform_thinking_blocks_with_redacted_content():
    thinking_block = {"redactedContent": "This is a redacted content"}
    decoder = AWSEventStreamDecoder(model="test")
    transformed_thinking_blocks = decoder.translate_thinking_blocks(thinking_block)
    assert len(transformed_thinking_blocks) == 1
    assert transformed_thinking_blocks[0]["type"] == "redacted_thinking"
    assert transformed_thinking_blocks[0]["data"] == "This is a redacted content"


def test_transform_tool_calls_index():
    chunks = [
        {
            "delta": {"text": "Certainly! I can help you with the"},
            "contentBlockIndex": 0,
        },
        {
            "delta": {"text": " current weather and time in Tokyo."},
            "contentBlockIndex": 0,
        },
        {"delta": {"text": " To get this information, I'll"}, "contentBlockIndex": 0},
        {"delta": {"text": " need to use two"}, "contentBlockIndex": 0},
        {"delta": {"text": " different tools: one"}, "contentBlockIndex": 0},
        {"delta": {"text": " for the weather and one for"}, "contentBlockIndex": 0},
        {"delta": {"text": " the time. Let me fetch"}, "contentBlockIndex": 0},
        {"delta": {"text": " that data for you."}, "contentBlockIndex": 0},
        {
            "start": {
                "toolUse": {
                    "toolUseId": "tooluse_JX1wqyUvRjyTcVSg_6-JwA",
                    "name": "Weather_Tool",
                }
            },
            "contentBlockIndex": 1,
        },
        {"delta": {"toolUse": {"input": ""}}, "contentBlockIndex": 1},
        {"delta": {"toolUse": {"input": '{"locatio'}}, "contentBlockIndex": 1},
        {"delta": {"toolUse": {"input": 'n": "Toky'}}, "contentBlockIndex": 1},
        {"delta": {"toolUse": {"input": 'o"}'}}, "contentBlockIndex": 1},
        {
            "start": {
                "toolUse": {
                    "toolUseId": "tooluse_rxDBNjDMQ-mqA-YOp9_3cQ",
                    "name": "Query_Time_Tool",
                }
            },
            "contentBlockIndex": 2,
        },
        {"delta": {"toolUse": {"input": ""}}, "contentBlockIndex": 2},
        {"delta": {"toolUse": {"input": '{"locati'}}, "contentBlockIndex": 2},
        {"delta": {"toolUse": {"input": 'on"'}}, "contentBlockIndex": 2},
        {"delta": {"toolUse": {"input": ': "Tokyo"}'}}, "contentBlockIndex": 2},
        {"stopReason": "tool_use"},
    ]
    decoder = AWSEventStreamDecoder(model="test")
    parsed_chunks = []
    for chunk in chunks:
        parsed_chunk = decoder._chunk_parser(chunk)
        parsed_chunks.append(parsed_chunk)
    tool_call_chunks1 = parsed_chunks[8:12]
    tool_call_chunks2 = parsed_chunks[13:17]
    for tool_call_hunk in tool_call_chunks1:
        tool_call_hunk_dict = tool_call_hunk.model_dump()
        for tool_call in tool_call_hunk_dict["choices"][0]["delta"]["tool_calls"]:
            assert tool_call["index"] == 0
    for tool_call_hunk in tool_call_chunks2:
        tool_call_hunk_dict = tool_call_hunk.model_dump()
        for tool_call in tool_call_hunk_dict["choices"][0]["delta"]["tool_calls"]:
            assert tool_call["index"] == 1


def test_transform_tool_calls_index_with_optional_arg_func():
    chunks = [
        {
            "contentBlockIndex": 0,
            "delta": {"text": "To"},
            "p": "abcdefghijklmnopqrstuv",
        },
        {
            "contentBlockIndex": 0,
            "delta": {"text": " get the current time, I"},
            "p": "abcdefghijklmnopqrstuvwxyzABCD",
        },
        {
            "contentBlockIndex": 0,
            "delta": {"text": ' can use the "get_time"'},
            "p": "abcdefghijkl",
        },
        {
            "contentBlockIndex": 0,
            "delta": {"text": " function. Since the user"},
            "p": "abcdefghijkl",
        },
        {
            "contentBlockIndex": 0,
            "delta": {"text": " didn't specify whether"},
            "p": "abcdefghijklmnopqrstuvw",
        },
        {
            "contentBlockIndex": 0,
            "delta": {"text": " they want UTC time or local time,"},
            "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV",
        },
        {
            "contentBlockIndex": 0,
            "delta": {"text": " I'll assume they"},
            "p": "abcdefghijkl",
        },
        {
            "contentBlockIndex": 0,
            "delta": {"text": " want the local time. Here's"},
            "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN",
        },
        {
            "contentBlockIndex": 0,
            "delta": {"text": " how I"},
            "p": "abcdefghijklmnopqrstuvw",
        },
        {
            "contentBlockIndex": 0,
            "delta": {"text": "'ll make the function call:"},
            "p": "abcdefghijklmnopqrstuvwxyzAB",
        },
        {
            "contentBlockIndex": 0,
            "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
        },
        {
            "contentBlockIndex": 1,
            "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNO",
            "start": {
                "toolUse": {
                    "name": "get_time",
                    "toolUseId": "tooluse_htgmgeJATsKTl4s_LW77sQ",
                }
            },
        },
        {
            "contentBlockIndex": 1,
            "delta": {"toolUse": {"input": ""}},
            "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV",
        },
        {"contentBlockIndex": 1, "p": "abcdefghijklmnopqrstuvw"},
        {"p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJK", "stopReason": "tool_use"},
    ]
    decoder = AWSEventStreamDecoder(model="test")
    parsed_chunks = []
    for chunk in chunks:
        parsed_chunk = decoder._chunk_parser(chunk)
        parsed_chunks.append(parsed_chunk)
    tool_call_chunks = parsed_chunks[11:14]
    for tool_call_hunk in tool_call_chunks:
        tool_call_hunk_dict = tool_call_hunk.model_dump()
        for tool_call in tool_call_hunk_dict["choices"][0]["delta"]["tool_calls"]:
            assert tool_call["index"] == 0


def test_bedrock_converse_streaming_consistent_id():
    """
    Tests that all chunks in a Bedrock Converse stream response share the same ID,
    capturing the ID from the initial 'messageStart' event.
    """
    # Simulate a realistic Bedrock Converse stream
    native_conversation_id = "a1b2c3d4-e5f6-7890-1234-56789abcdef0"
    mock_stream_chunks = [
        {
            "messageStart": {
                "conversationId": native_conversation_id,
                "role": "assistant",
            }
        },
        {"delta": {"text": "Hello"}, "contentBlockIndex": 0},
        {"delta": {"text": " world!"}, "contentBlockIndex": 0},
        {"stopReason": "stop"},
    ]

    decoder = AWSEventStreamDecoder(model="bedrock/anthropic.claude-3-sonnet-v1:0")

    # Process each chunk and collect the parsed responses
    parsed_responses = []
    for chunk in mock_stream_chunks:
        parsed_responses.append(decoder.converse_chunk_parser(chunk))

    # Verify that all parsed responses have the same, non-null ID derived from the native ID
    assert len(parsed_responses) > 1, "Should have processed multiple chunks"

    expected_id = f"chatcmpl-{native_conversation_id}"

    for response in parsed_responses:
        assert (
            response.id == expected_id
        ), "All chunk IDs must match the one captured from the messageStart event"


@pytest.mark.asyncio
async def test_make_call_does_not_rechunk_stream_by_default():
    """Re-chunking the event stream into fixed 1024-byte blocks holds small
    early events (messageStart, contentBlockStart) in httpx's ByteChunker until
    1024 bytes accumulate, delaying time-to-first-chunk by the whole generation
    when Bedrock trickles bytes (e.g. buffered tool-use streams)."""
    response = MagicMock()
    response.status_code = 200
    client = MagicMock()
    client.post = AsyncMock(return_value=response)

    await make_call(
        client=client,
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-6/converse-stream",
        headers={},
        data="{}",
        model="anthropic.claude-sonnet-4-6",
        messages=[],
        logging_obj=MagicMock(),
    )

    response.aiter_bytes.assert_called_once_with(chunk_size=None)


@pytest.mark.asyncio
async def test_make_call_honors_explicit_stream_chunk_size():
    response = MagicMock()
    response.status_code = 200
    client = MagicMock()
    client.post = AsyncMock(return_value=response)

    await make_call(
        client=client,
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-6/converse-stream",
        headers={},
        data="{}",
        model="anthropic.claude-sonnet-4-6",
        messages=[],
        logging_obj=MagicMock(),
        stream_chunk_size=2048,
    )

    response.aiter_bytes.assert_called_once_with(chunk_size=2048)


def test_make_sync_call_does_not_rechunk_stream_by_default():
    response = MagicMock()
    response.status_code = 200
    client = MagicMock()
    client.post = MagicMock(return_value=response)

    make_sync_call(
        client=client,
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-6/converse-stream",
        headers={},
        data="{}",
        signed_json_body=None,
        model="anthropic.claude-sonnet-4-6",
        messages=[],
        logging_obj=MagicMock(),
    )

    response.iter_bytes.assert_called_once_with(chunk_size=None)


def test_make_sync_call_honors_explicit_stream_chunk_size():
    response = MagicMock()
    response.status_code = 200
    client = MagicMock()
    client.post = MagicMock(return_value=response)

    make_sync_call(
        client=client,
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-6/converse-stream",
        headers={},
        data="{}",
        signed_json_body=None,
        model="anthropic.claude-sonnet-4-6",
        messages=[],
        logging_obj=MagicMock(),
        stream_chunk_size=2048,
    )

    response.iter_bytes.assert_called_once_with(chunk_size=2048)


def test_legacy_bedrock_llm_streaming_does_not_rechunk_by_default():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_bytes = MagicMock(return_value=iter([]))
    client = HTTPHandler()
    client.post = MagicMock(return_value=mock_response)

    BedrockLLM().completion(
        model="cohere.command-text-v14",
        messages=[{"role": "user", "content": "hi"}],
        api_base=None,
        custom_prompt_dict={},
        model_response=litellm.ModelResponse(),
        print_verbose=lambda *args, **kwargs: None,
        encoding=litellm.encoding,
        logging_obj=MagicMock(),
        optional_params={
            "stream": True,
            "aws_access_key_id": "fake",
            "aws_secret_access_key": "fake",
            "aws_region_name": "us-east-1",
        },
        acompletion=False,
        timeout=None,
        litellm_params={},
        client=client,
    )

    mock_response.iter_bytes.assert_called_once_with(chunk_size=None)
