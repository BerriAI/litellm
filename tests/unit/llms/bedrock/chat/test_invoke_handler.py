import base64
import binascii
import datetime
import itertools
import json
import re
import struct
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Final
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

import litellm
from litellm import AmazonInvokeConfig
from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.bedrock.chat.invoke_handler import (
    AmazonOpenAICompatibleStreamDecoder,
    AWSEventStreamDecoder,
    make_call,
    make_sync_call,
)
from litellm.llms.bedrock.common_utils import BedrockError, get_bedrock_stream_event_statuses
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.bedrock import BedrockInvokeNovaRequest
from litellm.types.utils import ModelResponseStream
from tests.unit.llms.bedrock.slow_upstream import (
    STREAM_TIMEOUT_SECONDS,
    slow_upstream_async_client,
    slow_upstream_sync_client,
)


@pytest.fixture
def bedrock_transformer() -> AmazonInvokeConfig:
    return AmazonInvokeConfig()


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
        parsed_chunk = decoder.chunk_parser(chunk)
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
        parsed_chunk = decoder.chunk_parser(chunk)
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
        assert response.id == expected_id, "All chunk IDs must match the one captured from the messageStart event"


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


def _converse_stream_wrapper(events, model=CONVERSE_MODEL):
    async def bedrock_stream():
        decoder = AWSEventStreamDecoder(model=model)
        for event in events:
            yield decoder.chunk_parser(chunk_data=event)

    return CustomStreamWrapper(
        completion_stream=bedrock_stream(),
        model=model,
        custom_llm_provider="bedrock",
        logging_obj=LiteLLMLoggingObj(
            model=model,
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
async def test_nova_invoke_stream_reports_bedrock_usage_and_finish_reason():
    """InvokeModel Nova wraps every Converse event under its event-type key and reports usage
    without ``totalTokens``; the stream must end on Bedrock's finish reason and surface the
    cached tokens instead of a token-count estimate."""
    events = (
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "OK"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "."}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {
                    "inputTokens": 5,
                    "outputTokens": 3,
                    "cacheReadInputTokenCount": 12262,
                    "cacheWriteInputTokenCount": 0,
                },
                "metrics": {},
                "trace": {},
            }
        },
    )
    wrapper = _converse_stream_wrapper(events, model="bedrock/invoke/us.amazon.nova-pro-v1:0")

    chunks = [chunk async for chunk in wrapper]

    assert "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices) == "OK."
    finish_reasons = [choice.finish_reason for chunk in chunks for choice in chunk.choices if choice.finish_reason]
    assert finish_reasons == ["stop"]
    assert chunks[-1].choices[0].finish_reason == "stop"
    usages = [chunk.usage for chunk in wrapper.chunks if getattr(chunk, "usage", None) is not None]
    assert len(usages) == 1
    assert usages[0].prompt_tokens == 12267
    assert usages[0].prompt_tokens_details.cached_tokens == 12262
    assert usages[0].completion_tokens == 3
    assert usages[0].total_tokens == 12270


def test_nova_invoke_remove_empty_system_messages():
    """Test that _remove_empty_system_messages removes empty system list."""
    input_request = BedrockInvokeNovaRequest(
        messages=[{"content": [{"text": "Hello"}], "role": "user"}],
        system=[],
        inferenceConfig={"temperature": 0.7},
    )

    litellm.AmazonInvokeNovaConfig()._remove_empty_system_messages(input_request)

    assert "system" not in input_request
    assert "messages" in input_request
    assert "inferenceConfig" in input_request


def test_nova_invoke_filter_allowed_fields():
    """
    Test that _filter_allowed_fields only keeps fields defined in BedrockInvokeNovaRequest.

    Nova Invoke does not allow `additionalModelRequestFields` and `additionalModelResponseFieldPaths` in the request body.
    This test ensures that these fields are not included in the request body.
    """
    _input_request = {
        "messages": [{"content": [{"text": "Hello"}], "role": "user"}],
        "system": [{"text": "System prompt"}],
        "inferenceConfig": {"temperature": 0.7},
        "additionalModelRequestFields": {"this": "should be removed"},
        "additionalModelResponseFieldPaths": ["this", "should", "be", "removed"],
    }

    input_request = BedrockInvokeNovaRequest(**_input_request)

    result = litellm.AmazonInvokeNovaConfig()._filter_allowed_fields(input_request)

    assert "additionalModelRequestFields" not in result
    assert "additionalModelResponseFieldPaths" not in result
    assert "messages" in result
    assert "system" in result
    assert "inferenceConfig" in result


def test_nova_invoke_streaming_chunk_parsing():
    """
    Test that the AWSEventStreamDecoder correctly handles Nova's /bedrock/invoke/ streaming format
    where content is nested under 'contentBlockDelta'.
    """
    from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

    # Initialize the decoder with a Nova model
    decoder = AWSEventStreamDecoder(model="bedrock/invoke/us.amazon.nova-micro-v1:0")

    # Test case 1: Text content in contentBlockDelta
    nova_text_chunk = {
        "contentBlockDelta": {
            "delta": {"text": "Hello, how can I help?"},
            "contentBlockIndex": 0,
        }
    }
    result = decoder.chunk_parser(nova_text_chunk)
    assert result.choices[0].delta.content == "Hello, how can I help?"
    assert result.choices[0].index == 0
    assert not result.choices[0].finish_reason
    assert result.choices[0].delta.tool_calls is None

    # Test case 2: Tool use start in contentBlockDelta
    nova_tool_start_chunk = {
        "contentBlockDelta": {
            "start": {"toolUse": {"name": "get_weather", "toolUseId": "tool_1"}},
            "contentBlockIndex": 1,
        }
    }
    result = decoder.chunk_parser(nova_tool_start_chunk)
    assert result.choices[0].delta.content == ""
    assert result.choices[0].index == 0
    assert result.choices[0].delta.tool_calls is not None
    assert result.choices[0].delta.tool_calls[0].type == "function"
    assert result.choices[0].delta.tool_calls[0].function.name == "get_weather"
    assert result.choices[0].delta.tool_calls[0].id == "tool_1"

    # Test case 3: Tool use arguments in contentBlockDelta
    nova_tool_args_chunk = {
        "contentBlockDelta": {
            "delta": {"toolUse": {"input": '{"location": "New York"}'}},
            "contentBlockIndex": 2,
        }
    }
    result = decoder.chunk_parser(nova_tool_args_chunk)
    assert result.choices[0].delta.content == ""
    assert result.choices[0].index == 0
    assert result.choices[0].delta.tool_calls is not None
    assert result.choices[0].delta.tool_calls[0].function.arguments == '{"location": "New York"}'

    # Test case 4: Stop reason in contentBlockDelta
    nova_stop_chunk = {
        "contentBlockDelta": {
            "stopReason": "tool_use",
        }
    }
    result = decoder.chunk_parser(nova_stop_chunk)
    print(result)
    assert result.choices[0].finish_reason == "tool_calls"


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


def _bedrock_stream_error_response(status_code: int, request_id: str) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={
            "x-amzn-RequestId": request_id,
            "x-amzn-ErrorType": "InternalServerException",
        },
        text='{"message":"Amazon Bedrock is unable to process your request."}',
        request=httpx.Request("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/"),
    )


def test_invoke_streaming_error_forwards_bedrock_response_headers():
    error_response = _bedrock_stream_error_response(500, "req-stream-err-1")
    client = HTTPHandler()
    client.post = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "server error",
            request=error_response.request,
            response=error_response,
        )
    )

    with pytest.raises(litellm.ServiceUnavailableError) as exc_info:
        litellm.completion(
            model="bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            client=client,
            aws_access_key_id="fake",
            aws_secret_access_key="fake",
            aws_region_name="us-east-1",
        )

    assert exc_info.value.response.headers["x-amzn-requestid"] == "req-stream-err-1"


@pytest.mark.asyncio
async def test_async_invoke_streaming_error_forwards_bedrock_response_headers():
    error_response = _bedrock_stream_error_response(500, "req-stream-err-2")
    client = AsyncHTTPHandler()
    client.post = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "server error",
            request=error_response.request,
            response=error_response,
        )
    )

    with pytest.raises(litellm.ServiceUnavailableError) as exc_info:
        await litellm.acompletion(
            model="bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            client=client,
            aws_access_key_id="fake",
            aws_secret_access_key="fake",
            aws_region_name="us-east-1",
        )

    assert exc_info.value.response.headers["x-amzn-requestid"] == "req-stream-err-2"


def _unread_bedrock_stream_error_response(status_code: int, request_id: str) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={
            "x-amzn-RequestId": request_id,
            "x-amzn-ErrorType": "InternalServerException",
        },
        stream=httpx.ByteStream(b'{"message":"Amazon Bedrock is unable to process your request."}'),
        request=httpx.Request("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/"),
    )


def test_invoke_streaming_error_forwards_headers_when_body_was_never_read():
    """A retried streamed request raises HTTPStatusError over a body nobody read, so
    reading it for the error message throws and loses the request id (LIT-5428)."""
    error_response = _unread_bedrock_stream_error_response(500, "req-unread-sync")
    client = HTTPHandler()
    client.post = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "server error",
            request=error_response.request,
            response=error_response,
        )
    )

    with pytest.raises(litellm.ServiceUnavailableError) as exc_info:
        litellm.completion(
            model="bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            client=client,
            aws_access_key_id="fake",
            aws_secret_access_key="fake",
            aws_region_name="us-east-1",
        )

    assert exc_info.value.response.headers["x-amzn-requestid"] == "req-unread-sync"


@pytest.mark.asyncio
async def test_async_invoke_streaming_error_forwards_headers_when_body_was_never_read():
    error_response = _unread_bedrock_stream_error_response(500, "req-unread-async")
    client = AsyncHTTPHandler()
    client.post = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "server error",
            request=error_response.request,
            response=error_response,
        )
    )

    with pytest.raises(litellm.ServiceUnavailableError) as exc_info:
        await litellm.acompletion(
            model="bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            client=client,
            aws_access_key_id="fake",
            aws_secret_access_key="fake",
            aws_region_name="us-east-1",
        )

    assert exc_info.value.response.headers["x-amzn-requestid"] == "req-unread-async"


def test_invoke_streaming_non_200_forwards_bedrock_response_headers():
    """A caller-supplied client that returns a failure instead of raising still reaches the
    provider's headers, and reading the streamed body for the message must not throw (LIT-5428)."""
    error_response = _unread_bedrock_stream_error_response(500, "req-non200-sync")
    client = HTTPHandler()
    client.post = MagicMock(return_value=error_response)

    with pytest.raises(litellm.ServiceUnavailableError) as exc_info:
        litellm.completion(
            model="bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            client=client,
            aws_access_key_id="fake",
            aws_secret_access_key="fake",
            aws_region_name="us-east-1",
        )

    assert exc_info.value.response.headers["x-amzn-requestid"] == "req-non200-sync"


@pytest.mark.asyncio
async def test_async_invoke_streaming_non_200_forwards_bedrock_response_headers():
    error_response = _unread_bedrock_stream_error_response(500, "req-non200-async")
    client = AsyncHTTPHandler()
    client.post = AsyncMock(return_value=error_response)

    with pytest.raises(litellm.ServiceUnavailableError) as exc_info:
        await litellm.acompletion(
            model="bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            client=client,
            aws_access_key_id="fake",
            aws_secret_access_key="fake",
            aws_region_name="us-east-1",
        )

    assert exc_info.value.response.headers["x-amzn-requestid"] == "req-non200-async"


def _event_stream_frame(event_type: str, payload: bytes) -> bytes:
    def header(name: str, value: str) -> bytes:
        return bytes([len(name)]) + name.encode() + bytes([7]) + struct.pack(">H", len(value)) + value.encode()

    headers: Final = (
        header(":event-type", event_type)
        + header(":content-type", "application/json")
        + header(":message-type", "event")
    )
    prelude: Final = struct.pack(">II", 12 + len(headers) + len(payload) + 4, len(headers))
    body: Final = prelude + struct.pack(">I", binascii.crc32(prelude)) + headers + payload
    return body + struct.pack(">I", binascii.crc32(body))


def _bedrock_event_stream_frame(chunk: Mapping[str, object]) -> bytes:
    return _event_stream_frame(
        "chunk", json.dumps({"bytes": base64.b64encode(json.dumps(chunk).encode()).decode()}).encode()
    )


def _converse_event_frame(event_type: str, body: Mapping[str, object]) -> bytes:
    return _event_stream_frame(event_type, json.dumps(body).encode())


def _openai_stream_chunk(delta: Mapping[str, str], finish_reason: str | None = None) -> Mapping[str, object]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "moonshot.kimi-k2-thinking",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


_MOONSHOT_RAW_STREAM: Final = b"".join(
    _bedrock_event_stream_frame(chunk)
    for chunk in (
        _openai_stream_chunk({"role": "assistant", "reasoning_content": "thinking"}),
        _openai_stream_chunk({"content": '{"city": '}),
        _openai_stream_chunk({"content": '"San Francisco"}'}),
        _openai_stream_chunk({}, "stop"),
    )
)


def _assert_moonshot_stream_content(chunks: Sequence[ModelResponseStream]) -> None:
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == '{"city": "San Francisco"}'
    assert "".join(getattr(chunk.choices[0].delta, "reasoning_content", None) or "" for chunk in chunks) == "thinking"
    assert [chunk.choices[0].finish_reason for chunk in chunks if chunk.choices[0].finish_reason] == ["stop"]


@pytest.fixture
def _aws_test_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")


@pytest.mark.parametrize("response_format", [None, {"type": "json_object"}])
def test_moonshot_invoke_stream_yields_openai_shaped_chunks(
    _aws_test_credentials: None, response_format: Mapping[str, str] | None
) -> None:
    raw_stream: Final = _MOONSHOT_RAW_STREAM
    response: Final = MagicMock(status_code=200, headers={})
    response.iter_bytes = lambda chunk_size=None: iter([raw_stream])
    client: Final = HTTPHandler()
    client.post = MagicMock(return_value=response)

    stream: Final = litellm.completion(
        model="bedrock/invoke/moonshot.kimi-k2-thinking",
        messages=[{"role": "user", "content": "weather as json"}],
        stream=True,
        client=client,
        **({"response_format": response_format} if response_format else {}),
    )
    _assert_moonshot_stream_content(list(stream))


@pytest.mark.asyncio
async def test_moonshot_invoke_async_stream_yields_openai_shaped_chunks(_aws_test_credentials: None) -> None:
    async def _aiter_bytes(chunk_size: int | None = None) -> AsyncIterator[bytes]:
        yield _MOONSHOT_RAW_STREAM

    response: Final = MagicMock(status_code=200, headers={})
    response.aiter_bytes = _aiter_bytes
    client: Final = AsyncHTTPHandler()
    client.post = AsyncMock(return_value=response)

    stream: Final = await litellm.acompletion(
        model="bedrock/invoke/moonshot.kimi-k2-thinking",
        messages=[{"role": "user", "content": "weather as json"}],
        stream=True,
        response_format={"type": "json_object"},
        client=client,
    )

    _assert_moonshot_stream_content([chunk async for chunk in stream])


def _truncated_frame() -> bytes:
    return _bedrock_event_stream_frame(_openai_stream_chunk({"role": "assistant"}))[:-8]


def _event_stream_headers() -> httpx.Headers:
    return httpx.Headers({"content-type": "application/vnd.amazon.eventstream", "x-amzn-RequestId": "req-empty-1"})


_UNDECODABLE_STREAM_BODIES: Final = (
    pytest.param(b"", id="empty"),
    pytest.param(b"\x00\x00\x00\x05", id="shorter-than-a-prelude"),
    pytest.param(_truncated_frame(), id="truncated-first-message"),
)


def _assert_no_events_error(error: BedrockError, body: bytes) -> None:
    assert error.status_code == 502
    assert "HTTP 200" in error.message
    assert "decoded to no events" in error.message
    assert f"{len(body)} bytes received" in error.message
    assert "application/vnd.amazon.eventstream" in error.message
    assert "req-empty-1" in error.message
    assert f"first bytes={body[:200]!r}" in error.message


@pytest.mark.parametrize("body", _UNDECODABLE_STREAM_BODIES)
def test_iter_bytes_raises_when_a_200_body_decodes_to_no_events(body: bytes) -> None:
    decoder: Final = AWSEventStreamDecoder(model="us.moonshotai.kimi-k3")

    with pytest.raises(BedrockError) as exc_info:
        list(decoder.iter_bytes(iter([body]), response_headers=_event_stream_headers()))

    _assert_no_events_error(exc_info.value, body)


@pytest.mark.asyncio
@pytest.mark.parametrize("body", _UNDECODABLE_STREAM_BODIES)
async def test_aiter_bytes_raises_when_a_200_body_decodes_to_no_events(body: bytes) -> None:
    async def _chunks() -> AsyncIterator[bytes]:
        yield body

    decoder: Final = AWSEventStreamDecoder(model="us.moonshotai.kimi-k3")

    with pytest.raises(BedrockError) as exc_info:
        _ = [chunk async for chunk in decoder.aiter_bytes(_chunks(), response_headers=_event_stream_headers())]

    _assert_no_events_error(exc_info.value, body)


def test_iter_bytes_raises_when_the_stream_ends_mid_message() -> None:
    decoder: Final = AmazonOpenAICompatibleStreamDecoder(model="moonshot.kimi-k2-thinking", sync_stream=True)
    stream: Final = decoder.iter_bytes(iter([_MOONSHOT_RAW_STREAM, _truncated_frame()]))

    chunks: Final = list(itertools.islice(stream, 4))
    with pytest.raises(BedrockError) as exc_info:
        next(stream)

    _assert_moonshot_stream_content(chunks)
    assert exc_info.value.status_code == 502
    assert f"{len(_truncated_frame())} undecoded bytes after 4 events" in exc_info.value.message
    assert "first bytes=" not in exc_info.value.message


def test_iter_bytes_yields_a_complete_stream_without_raising() -> None:
    decoder: Final = AmazonOpenAICompatibleStreamDecoder(model="moonshot.kimi-k2-thinking", sync_stream=True)

    chunks: Final = list(decoder.iter_bytes(iter([_MOONSHOT_RAW_STREAM[:100], _MOONSHOT_RAW_STREAM[100:]])))

    _assert_moonshot_stream_content(chunks)


def _assert_empty_stream_surfaced_as_bad_gateway(error: MidStreamFallbackError) -> None:
    assert error.status_code == 502
    assert error.is_pre_first_chunk is True
    assert isinstance(error.original_exception, litellm.BadGatewayError)
    assert "decoded to no events" in str(error)
    assert "req-empty-1" in str(error)


def test_converse_stream_with_an_empty_200_body_raises_instead_of_an_empty_turn(_aws_test_credentials: None) -> None:
    response: Final = MagicMock(status_code=200, headers=_event_stream_headers())
    response.iter_bytes = lambda chunk_size=None: iter([b""])
    client: Final = HTTPHandler()
    client.post = MagicMock(return_value=response)

    with pytest.raises(MidStreamFallbackError) as exc_info:
        list(
            litellm.completion(
                model="bedrock/us.moonshotai.kimi-k3",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                client=client,
            )
        )

    _assert_empty_stream_surfaced_as_bad_gateway(exc_info.value)


@pytest.mark.asyncio
async def test_async_converse_stream_with_an_empty_200_body_raises_instead_of_an_empty_turn(
    _aws_test_credentials: None,
) -> None:
    async def _aiter_bytes(chunk_size: int | None = None) -> AsyncIterator[bytes]:
        yield b""

    response: Final = MagicMock(status_code=200, headers=_event_stream_headers())
    response.aiter_bytes = _aiter_bytes
    client: Final = AsyncHTTPHandler()
    client.post = AsyncMock(return_value=response)

    stream: Final = await litellm.acompletion(
        model="bedrock/us.moonshotai.kimi-k3",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=client,
    )
    with pytest.raises(MidStreamFallbackError) as exc_info:
        _ = [chunk async for chunk in stream]

    _assert_empty_stream_surfaced_as_bad_gateway(exc_info.value)


_UPSTREAM_REJECTION: Final = "structured output schema uses unsupported regex negative look-ahead"
_CUSTOMER_REJECTION_EVENT_TYPE: Final = "validationException"


def _modeled_exception_event_types() -> tuple[str, ...]:
    statuses: Final = get_bedrock_stream_event_statuses()
    assert statuses is not None
    return tuple(sorted(name for name, status in statuses.items() if status is not None))


def _modeled_status(event_type: str) -> int:
    statuses: Final = get_bedrock_stream_event_statuses()
    assert statuses is not None
    status: Final = statuses[event_type]
    assert status is not None
    return status


_CONVERSE_CONTENT_FRAMES: Final = (
    _converse_event_frame("messageStart", {"role": "assistant"}),
    _converse_event_frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "hi"}}),
    _converse_event_frame("contentBlockStop", {"contentBlockIndex": 0}),
    _converse_event_frame("messageStop", {"stopReason": "end_turn"}),
)


def _unknown_event_frame() -> bytes:
    return _converse_event_frame("somethingBedrockAddedLater", {"message": _UPSTREAM_REJECTION})


@pytest.mark.parametrize("event_type", _modeled_exception_event_types())
def test_iter_bytes_raises_the_modeled_error_for_an_exception_named_event_frame(event_type: str) -> None:
    decoder: Final = AWSEventStreamDecoder(model="us.moonshotai.kimi-k3")
    frame: Final = _converse_event_frame(event_type, {"message": _UPSTREAM_REJECTION})

    with pytest.raises(BedrockError) as exc_info:
        list(decoder.iter_bytes(iter([frame]), response_headers=_event_stream_headers()))

    assert exc_info.value.status_code == _modeled_status(event_type)
    assert exc_info.value.status_code != 200
    assert exc_info.value.message.startswith(event_type)
    assert _UPSTREAM_REJECTION in exc_info.value.message


@pytest.mark.asyncio
async def test_aiter_bytes_raises_the_modeled_error_for_an_exception_named_event_frame() -> None:
    event_type: Final = _CUSTOMER_REJECTION_EVENT_TYPE

    async def _chunks() -> AsyncIterator[bytes]:
        yield _converse_event_frame(event_type, {"message": _UPSTREAM_REJECTION})

    decoder: Final = AWSEventStreamDecoder(model="us.moonshotai.kimi-k3")

    with pytest.raises(BedrockError) as exc_info:
        _ = [chunk async for chunk in decoder.aiter_bytes(_chunks(), response_headers=_event_stream_headers())]

    assert exc_info.value.status_code == _modeled_status(event_type)
    assert _UPSTREAM_REJECTION in exc_info.value.message


def _assert_unknown_event_stream_error(error: BedrockError, body: bytes) -> None:
    assert error.status_code == 502
    assert "HTTP 200" in error.message
    assert "none of its 1 events carried a known event type" in error.message
    assert "somethingBedrockAddedLater" in error.message
    assert _UPSTREAM_REJECTION in error.message
    assert f"{len(body)} bytes received" in error.message
    assert "req-empty-1" in error.message


def test_iter_bytes_raises_when_no_event_carries_a_known_event_type() -> None:
    decoder: Final = AWSEventStreamDecoder(model="us.moonshotai.kimi-k3")
    body: Final = _unknown_event_frame()

    with pytest.raises(BedrockError) as exc_info:
        list(decoder.iter_bytes(iter([body]), response_headers=_event_stream_headers()))

    _assert_unknown_event_stream_error(exc_info.value, body)


@pytest.mark.asyncio
async def test_aiter_bytes_raises_when_no_event_carries_a_known_event_type() -> None:
    body: Final = _unknown_event_frame()

    async def _chunks() -> AsyncIterator[bytes]:
        yield body

    decoder: Final = AWSEventStreamDecoder(model="us.moonshotai.kimi-k3")

    with pytest.raises(BedrockError) as exc_info:
        _ = [chunk async for chunk in decoder.aiter_bytes(_chunks(), response_headers=_event_stream_headers())]

    _assert_unknown_event_stream_error(exc_info.value, body)


def test_iter_bytes_keeps_a_stream_whose_unknown_event_sits_beside_known_frames() -> None:
    decoder: Final = AWSEventStreamDecoder(model="us.moonshotai.kimi-k3")
    frames: Final = (_CONVERSE_CONTENT_FRAMES[0], _unknown_event_frame(), *_CONVERSE_CONTENT_FRAMES[1:])

    chunks: Final = list(decoder.iter_bytes(iter(frames), response_headers=_event_stream_headers()))

    texts: Final = [chunk.choices[0].delta.content for chunk in chunks if isinstance(chunk, ModelResponseStream)]
    assert "".join(text or "" for text in texts) == "hi"
    finish_reasons: Final = [
        chunk.choices[0].finish_reason for chunk in chunks if isinstance(chunk, ModelResponseStream)
    ]
    assert "stop" in finish_reasons


def _assert_exception_event_surfaced_with_its_modeled_status(error: BaseException, event_type: str) -> None:
    assert not isinstance(error, litellm.BadGatewayError)
    assert getattr(error, "status_code", None) == _modeled_status(event_type)
    assert event_type in str(error)
    assert _UPSTREAM_REJECTION in str(error)


def test_converse_stream_with_an_exception_event_frame_raises_instead_of_an_empty_turn(
    _aws_test_credentials: None,
) -> None:
    event_type: Final = _CUSTOMER_REJECTION_EVENT_TYPE
    frame: Final = _converse_event_frame(event_type, {"message": _UPSTREAM_REJECTION})
    response: Final = MagicMock(status_code=200, headers=_event_stream_headers())
    response.iter_bytes = lambda chunk_size=None: iter([frame])
    client: Final = HTTPHandler()
    client.post = MagicMock(return_value=response)

    with pytest.raises(Exception, match=re.escape(_UPSTREAM_REJECTION)) as exc_info:
        list(
            litellm.completion(
                model="bedrock/us.moonshotai.kimi-k3",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                client=client,
            )
        )

    _assert_exception_event_surfaced_with_its_modeled_status(exc_info.value, event_type)


@pytest.mark.asyncio
async def test_async_converse_stream_with_an_exception_event_frame_raises_instead_of_an_empty_turn(
    _aws_test_credentials: None,
) -> None:
    event_type: Final = _CUSTOMER_REJECTION_EVENT_TYPE

    async def _aiter_bytes(chunk_size: int | None = None) -> AsyncIterator[bytes]:
        yield _converse_event_frame(event_type, {"message": _UPSTREAM_REJECTION})

    response: Final = MagicMock(status_code=200, headers=_event_stream_headers())
    response.aiter_bytes = _aiter_bytes
    client: Final = AsyncHTTPHandler()
    client.post = AsyncMock(return_value=response)

    stream: Final = await litellm.acompletion(
        model="bedrock/us.moonshotai.kimi-k3",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=client,
    )
    with pytest.raises(Exception, match=re.escape(_UPSTREAM_REJECTION)) as exc_info:
        _ = [chunk async for chunk in stream]

    _assert_exception_event_surfaced_with_its_modeled_status(exc_info.value, event_type)


def test_converse_stream_made_only_of_unknown_events_raises_instead_of_an_empty_turn(
    _aws_test_credentials: None,
) -> None:
    response: Final = MagicMock(status_code=200, headers=_event_stream_headers())
    response.iter_bytes = lambda chunk_size=None: iter([_unknown_event_frame()])
    client: Final = HTTPHandler()
    client.post = MagicMock(return_value=response)

    with pytest.raises(MidStreamFallbackError) as exc_info:
        list(
            litellm.completion(
                model="bedrock/us.moonshotai.kimi-k3",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                client=client,
            )
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.is_pre_first_chunk is True
    assert isinstance(exc_info.value.original_exception, litellm.BadGatewayError)
    assert "somethingBedrockAddedLater" in str(exc_info.value)
    assert _UPSTREAM_REJECTION in str(exc_info.value)


def _invoke_streaming_kwargs() -> dict[str, object]:
    return {
        "model": "bedrock/invoke/anthropic.claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "timeout": STREAM_TIMEOUT_SECONDS,
        "aws_access_key_id": "fake",
        "aws_secret_access_key": "fake",
        "aws_region_name": "us-east-1",
    }


@pytest.mark.asyncio
async def test_async_invoke_streaming_fails_at_the_request_timeout_not_the_upstreams_pace() -> None:
    with pytest.raises(litellm.Timeout):
        await litellm.acompletion(client=slow_upstream_async_client(), **_invoke_streaming_kwargs())


def test_sync_invoke_streaming_fails_at_the_request_timeout_not_the_upstreams_pace() -> None:
    with pytest.raises(litellm.Timeout):
        litellm.completion(client=slow_upstream_sync_client(), **_invoke_streaming_kwargs())


def test_get_complete_url_basic(bedrock_transformer):
    """Test basic URL construction for non-streaming request"""
    url = bedrock_transformer.get_complete_url(
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com",
        api_key=None,
        model="anthropic.claude-v2",
        optional_params={},
        stream=False,
        litellm_params={},
    )
    assert url == "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/invoke"


def test_get_complete_url_streaming(bedrock_transformer):
    """Test URL construction for streaming request"""
    url = bedrock_transformer.get_complete_url(
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com",
        api_key=None,
        model="anthropic.claude-v2",
        optional_params={},
        stream=True,
        litellm_params={},
    )
    assert (
        url == "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/invoke-with-response-stream"
    )


def test_transform_request_invalid_provider(bedrock_transformer):
    """Test request transformation with invalid provider"""
    messages = [{"role": "user", "content": "Hello"}]
    with pytest.raises(Exception, match="Bedrock Invoke HTTPX: Unknown provider=None") as exc_info:
        bedrock_transformer.transform_request(
            model="invalid.model", messages=messages, optional_params={}, litellm_params={}, headers={}
        )
    assert "Unknown provider" in str(exc_info.value)


@patch("botocore.auth.SigV4Auth")
@patch("botocore.awsrequest.AWSRequest")
def test_sign_request_basic(mock_aws_request, mock_sigv4_auth, bedrock_transformer):
    """Test basic request signing without extra headers"""
    mock_credentials = Mock()
    bedrock_transformer.get_credentials = Mock(return_value=mock_credentials)
    mock_auth_instance = Mock()
    mock_sigv4_auth.return_value = mock_auth_instance
    mock_request = Mock()
    mock_request.headers = {
        "Authorization": "AWS4-HMAC-SHA256 Credential=...",
        "X-Amz-Date": "20240101T000000Z",
        "Content-Type": "application/json",
    }
    mock_aws_request.return_value = mock_request
    headers = {}
    optional_params = {"aws_region_name": "us-east-1"}
    request_data = {"prompt": "Hello"}
    api_base = "https://bedrock-runtime.us-east-1.amazonaws.com"
    result, _ = bedrock_transformer.sign_request(
        headers=headers, optional_params=optional_params, request_data=request_data, api_base=api_base
    )
    mock_sigv4_auth.assert_called_once_with(mock_credentials, "bedrock", "us-east-1")
    mock_aws_request.assert_called_once_with(
        method="POST", url=api_base, data='{"prompt": "Hello"}', headers={"Content-Type": "application/json"}
    )
    mock_auth_instance.add_auth.assert_called_once_with(mock_request)
    assert result == mock_request.headers


def test_transform_request_cohere_command(bedrock_transformer):
    """Test request transformation for Cohere Command model"""
    messages = [{"role": "user", "content": "Hello"}]
    result = bedrock_transformer.transform_request(
        model="cohere.command-r", messages=messages, optional_params={"max_tokens": 2048}, litellm_params={}, headers={}
    )
    print("transformed request for invoke cohere command=", json.dumps(result, indent=4))
    expected_result = {"message": "Hello", "max_tokens": 2048, "chat_history": []}
    assert result == expected_result


def test_transform_request_ai21(bedrock_transformer):
    """Test request transformation for AI21"""
    messages = [{"role": "user", "content": "Hello"}]
    result = bedrock_transformer.transform_request(
        model="ai21.j2-ultra", messages=messages, optional_params={"max_tokens": 2048}, litellm_params={}, headers={}
    )
    print("transformed request for invoke ai21=", json.dumps(result, indent=4))
    expected_result = {"prompt": "Hello", "max_tokens": 2048}
    assert result == expected_result


def test_transform_request_mistral(bedrock_transformer):
    """Test request transformation for Mistral"""
    messages = [{"role": "user", "content": "Hello"}]
    result = bedrock_transformer.transform_request(
        model="mistral.mistral-7b",
        messages=messages,
        optional_params={"max_tokens": 2048},
        litellm_params={},
        headers={},
    )
    print("transformed request for invoke mistral=", json.dumps(result, indent=4))
    expected_result = {"prompt": "<s>[INST] Hello [/INST]\n", "max_tokens": 2048}
    assert result == expected_result


def test_transform_request_amazon_titan(bedrock_transformer):
    """Test request transformation for Amazon Titan"""
    messages = [{"role": "user", "content": "Hello"}]
    result = bedrock_transformer.transform_request(
        model="amazon.titan-text-express-v1",
        messages=messages,
        optional_params={"maxTokenCount": 2048},
        litellm_params={},
        headers={},
    )
    print("transformed request for invoke amazon titan=", json.dumps(result, indent=4))
    expected_result = {"inputText": "\n\nUser: Hello\n\nBot: ", "textGenerationConfig": {"maxTokenCount": 2048}}
    assert result == expected_result


def test_filter_headers_for_aws_signature():
    """Test that header filtering works correctly for AWS signature calculation"""
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

    aws_llm = BaseAWSLLM()
    test_headers = {
        "Content-Type": "application/json",
        "Host": "bedrock-runtime.us-east-1.amazonaws.com",
        "x-amz-date": "20240101T120000Z",
        "x-amz-security-token": "test-token",
        "x-custom-header": "custom-value",
        "x-litellm-user-id": "user123",
        "x-forwarded-for": "192.168.1.1",
        "authorization": "Bearer test-token",
        "user-agent": "test-agent",
        "x-envoy-expected-rq-timeout-ms": "300000",
        "x-envoy-external-address": "10.105.1.156",
    }
    filtered_headers = aws_llm._filter_headers_for_aws_signature(test_headers)
    expected_aws_headers = {
        "Content-Type": "application/json",
        "Host": "bedrock-runtime.us-east-1.amazonaws.com",
        "x-amz-date": "20240101T120000Z",
        "x-amz-security-token": "test-token",
    }
    assert filtered_headers == expected_aws_headers, f"Expected {expected_aws_headers}, got {filtered_headers}"
    excluded_headers = [
        "x-custom-header",
        "x-litellm-user-id",
        "x-forwarded-for",
        "user-agent",
        "x-envoy-expected-rq-timeout-ms",
        "x-envoy-external-address",
    ]
    for header in excluded_headers:
        assert header not in filtered_headers, f"Header {header} should not be in filtered headers"
    empty_filtered = aws_llm._filter_headers_for_aws_signature({})
    assert empty_filtered == {}
    non_aws_headers = {"x-custom-trace": "trace-123", "x-user-context": "premium", "x-request-source": "mobile-app"}
    filtered_non_aws = aws_llm._filter_headers_for_aws_signature(non_aws_headers)
    assert filtered_non_aws == {}
