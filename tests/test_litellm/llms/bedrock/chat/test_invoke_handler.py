import datetime
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.bedrock.chat.invoke_handler import (
    AWSEventStreamDecoder,
    make_call,
    make_sync_call,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


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


def test_converse_streaming_usage_uses_provider_thinking_tokens():
    """Regression LIT-5714: the messageStop event carries provider thinking tokens
    under ``additionalModelResponseFields``; the usage chunk must report them instead
    of a token_counter estimate."""
    chunks = [
        {
            "contentBlockIndex": 0,
            "delta": {"reasoningContent": {"text": "thinking about it"}},
        },
        {
            "stopReason": "end_turn",
            "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 1033}}},
        },
        {"usage": {"inputTokens": 40, "outputTokens": 3002, "totalTokens": 3042}},
    ]

    decoder = AWSEventStreamDecoder(model="bedrock/anthropic.claude-opus-4-7")
    parsed = [decoder.converse_chunk_parser(chunk) for chunk in chunks]

    usage = parsed[-1].usage
    assert usage.completion_tokens_details.reasoning_tokens == 1033


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


CONVERSE_MODEL = "anthropic.claude-sonnet-4-6"
CONVERSE_METADATA_EVENT = {
    "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    "metrics": {"latencyMs": 100},
}


def _converse_stream_wrapper(events):
    async def bedrock_stream():
        decoder = AWSEventStreamDecoder(model=CONVERSE_MODEL)
        for event in events:
            yield decoder._chunk_parser(chunk_data=event)

    return CustomStreamWrapper(
        completion_stream=bedrock_stream(),
        model=CONVERSE_MODEL,
        custom_llm_provider="bedrock",
        logging_obj=LiteLLMLoggingObj(
            model=CONVERSE_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            call_type="completion",
            start_time=datetime.datetime.now(),
            litellm_call_id="1234",
            function_id="1234",
        ),
    )


@pytest.mark.parametrize(
    "events, expected_finish_reason",
    [
        pytest.param(
            (
                {"role": "assistant"},
                {"contentBlockIndex": 0, "delta": {"text": "Hello"}},
                {"contentBlockIndex": 0, "delta": {"text": " world"}},
                {"contentBlockIndex": 0},
                {"stopReason": "end_turn"},
                CONVERSE_METADATA_EVENT,
            ),
            "stop",
            id="text",
        ),
        pytest.param(
            (
                {"role": "assistant"},
                {"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "t1", "name": "get_weather"}}},
                {"contentBlockIndex": 0, "delta": {"toolUse": {"input": '{"city": "SF"}'}}},
                {"contentBlockIndex": 0},
                {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "t2", "name": "get_time"}}},
                {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"tz": "PT"}'}}},
                {"contentBlockIndex": 1},
                {"stopReason": "tool_use"},
                CONVERSE_METADATA_EVENT,
            ),
            "tool_calls",
            id="multiple_tool_calls",
        ),
        pytest.param(
            (
                {"role": "assistant"},
                {"contentBlockIndex": 0, "start": {}},
                {"contentBlockIndex": 0, "delta": {"text": "Let me check."}},
                {"contentBlockIndex": 0},
                {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "t1", "name": "get_weather"}}},
                {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"city": "SF"}'}}},
                {"contentBlockIndex": 1},
                {"stopReason": "tool_use"},
                CONVERSE_METADATA_EVENT,
            ),
            "tool_calls",
            id="text_then_tool_call",
        ),
        pytest.param(
            (
                {"role": "assistant"},
                {"contentBlockIndex": 0, "start": {}},
                {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "thinking hard"}}},
                {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig123"}}},
                {"contentBlockIndex": 0},
                {"contentBlockIndex": 1, "start": {}},
                {"contentBlockIndex": 1, "delta": {"text": "Answer"}},
                {"contentBlockIndex": 1},
                {"stopReason": "end_turn"},
                CONVERSE_METADATA_EVENT,
            ),
            "stop",
            id="reasoning_then_text",
        ),
    ],
)
@pytest.mark.asyncio
async def test_converse_stream_ends_on_finish_reason_chunk(events, expected_finish_reason):
    """The usage-only metadata event Bedrock sends after messageStop must not reach the caller as an extra
    assistant delta following the finish_reason chunk."""
    wrapper = _converse_stream_wrapper(events)

    chunks = [chunk async for chunk in wrapper]

    finish_reasons = [choice.finish_reason for chunk in chunks for choice in chunk.choices if choice.finish_reason]
    assert finish_reasons == [expected_finish_reason]
    assert chunks[-1].choices[0].finish_reason == expected_finish_reason, (
        f"stream must end on the finish_reason chunk, got trailing {chunks[-1].model_dump(exclude_none=True)}"
    )
    roles = [choice.delta.role for chunk in chunks for choice in chunk.choices if choice.delta.role]
    assert roles == ["assistant"]
    assert any(getattr(chunk, "usage", None) is not None for chunk in wrapper.chunks)


@pytest.mark.asyncio
async def test_converse_stream_still_emits_guardrail_trace_after_finish_reason():
    """Guardrail metadata events carry a trace payload alongside usage; that chunk must still reach the caller
    after the finish_reason chunk, as it did before the regression."""
    trace = {"guardrail": {"inputAssessment": {"g1": {}}}}
    events = (
        {"role": "assistant"},
        {"contentBlockIndex": 0, "delta": {"text": "Hello"}},
        {"contentBlockIndex": 0},
        {"stopReason": "end_turn"},
        {**CONVERSE_METADATA_EVENT, "trace": trace},
    )
    wrapper = _converse_stream_wrapper(events)

    chunks = [chunk async for chunk in wrapper]

    finish_reasons = [choice.finish_reason for chunk in chunks for choice in chunk.choices if choice.finish_reason]
    assert finish_reasons == ["stop"]
    assert chunks[-1].provider_specific_fields == {"trace": trace}
    assert chunks[-1].choices[0].delta.content == ""
    assert chunks[-1].choices[0].delta.role == "assistant"


def test_invoke_streaming_forwards_bedrock_response_headers():
    response = MagicMock()
    response.status_code = 200
    response.iter_bytes = MagicMock(return_value=iter([]))
    response.headers = httpx.Headers({"x-amzn-requestid": "req-789"})
    client = HTTPHandler()
    client.post = MagicMock(return_value=response)

    stream = litellm.completion(
        model="bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region_name="us-east-1",
    )

    assert stream._hidden_params["additional_headers"]["llm_provider-x-amzn-requestid"] == "req-789"


@pytest.mark.asyncio
async def test_async_invoke_streaming_forwards_bedrock_response_headers():
    async def _no_bytes(chunk_size=None):
        return
        yield b""

    response = MagicMock()
    response.status_code = 200
    response.aiter_bytes = _no_bytes
    response.headers = httpx.Headers({"x-amzn-requestid": "req-987"})
    client = AsyncHTTPHandler()
    client.post = AsyncMock(return_value=response)

    stream = await litellm.acompletion(
        model="bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region_name="us-east-1",
    )

    assert stream._hidden_params["additional_headers"]["llm_provider-x-amzn-requestid"] == "req-987"

