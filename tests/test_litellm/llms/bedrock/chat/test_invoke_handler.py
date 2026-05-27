import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder


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


# -----------------------------------------------------------------------------
# LIT-3274 -- defensive event-stream error surfacing
#
# When Bedrock returns a plain JSON error body (malformed URL paths, throttling,
# transient validation errors) in place of the binary event-stream framing,
# botocore.eventstream.EventStreamBuffer used to raise ChecksumMismatch on the
# JSON prelude, masking the real Bedrock error message from the caller.
#
# ``AWSEventStreamDecoder._maybe_surface_json_error`` now sniffs the accumulated
# payload and surfaces the underlying error when the bytes are clearly JSON.
# When they are not, the original ChecksumMismatch is preserved unchanged so
# we never widen the error surface beyond clearly-JSON payloads.
# -----------------------------------------------------------------------------
import asyncio  # noqa: E402

import pytest  # noqa: E402

from litellm.llms.bedrock.common_utils import BedrockError  # noqa: E402


def _drive_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_iter_bytes_surfaces_json_error_body():
    """Sync iter_bytes: JSON error body -> BedrockError with the real message."""
    decoder = AWSEventStreamDecoder(model="bedrock/anthropic.claude")
    payload = (
        b'{"message": "The provided model identifier is invalid.", '
        b'"code": "ValidationException"}'
    )
    with pytest.raises(BedrockError) as exc:
        list(decoder.iter_bytes(iter([payload])))
    assert "The provided model identifier is invalid." in exc.value.message
    assert exc.value.status_code == 400


def test_aiter_bytes_surfaces_json_error_body():
    """Async aiter_bytes: JSON error body -> BedrockError with the real message."""

    async def _producer():
        yield b'{"message": "Throttling exception", "code": "ThrottlingException"}'

    async def _run():
        decoder = AWSEventStreamDecoder(model="bedrock/anthropic.claude")
        with pytest.raises(BedrockError) as exc:
            async for _ in decoder.aiter_bytes(_producer()):
                pass
        return exc.value

    err = _drive_async(_run())
    assert "Throttling exception" in err.message
    assert err.status_code == 400


def test_iter_bytes_surfaces_capitalised_message_field():
    """Bedrock sometimes returns ``Message`` (capital M) -- handle both."""
    decoder = AWSEventStreamDecoder(model="bedrock/anthropic.claude")
    payload = b'{"Message": "The supplied authentication is not authorized."}'
    with pytest.raises(BedrockError) as exc:
        list(decoder.iter_bytes(iter([payload])))
    assert "The supplied authentication is not authorized." in exc.value.message


def test_iter_bytes_falls_back_to_json_dump_when_message_missing():
    """Unknown JSON shape: surface the payload, do not drop it silently."""
    decoder = AWSEventStreamDecoder(model="bedrock/anthropic.claude")
    payload = b'{"unexpected_field": "no recognised message key"}'
    with pytest.raises(BedrockError) as exc:
        list(decoder.iter_bytes(iter([payload])))
    assert "unexpected_field" in exc.value.message
    assert "no recognised message key" in exc.value.message


def test_iter_bytes_preserves_checksum_mismatch_for_non_json_garbage():
    """Non-JSON garbage must still raise the original ChecksumMismatch -- we
    never widen the error surface beyond clearly-JSON payloads."""
    from botocore.eventstream import ChecksumMismatch

    decoder = AWSEventStreamDecoder(model="bedrock/anthropic.claude")
    garbage = b"\x00\x00\x00\x10\x00\x00\x00\x04\xde\xad\xbe\xef\xca\xfe\xba\xbe"
    with pytest.raises(ChecksumMismatch):
        list(decoder.iter_bytes(iter([garbage])))


def test_iter_bytes_preserves_checksum_mismatch_when_payload_is_truncated_json():
    """Truncated JSON should not masquerade as a Bedrock error -- re-raise the
    original ChecksumMismatch so the caller can treat it as a transport
    failure rather than an authoritative error."""
    from botocore.eventstream import ChecksumMismatch

    decoder = AWSEventStreamDecoder(model="bedrock/anthropic.claude")
    truncated = b'{"message": "this gets cut'
    with pytest.raises(ChecksumMismatch):
        list(decoder.iter_bytes(iter([truncated])))


def test_maybe_surface_json_error_helper_preserves_original_for_empty_input():
    """Empty bytes must re-raise the original exception."""
    from botocore.eventstream import ChecksumMismatch

    original = ChecksumMismatch(expected=0, calculated=1)
    with pytest.raises(ChecksumMismatch):
        AWSEventStreamDecoder._maybe_surface_json_error(b"", original)


def test_maybe_surface_json_error_helper_surfaces_array_payload():
    """JSON array body is unusual but should still be surfaced."""
    from botocore.eventstream import ChecksumMismatch

    original = ChecksumMismatch(expected=0, calculated=1)
    with pytest.raises(BedrockError) as exc:
        AWSEventStreamDecoder._maybe_surface_json_error(b'[{"foo": 1}]', original)
    assert "foo" in exc.value.message
