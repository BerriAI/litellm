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


# ---------------------------------------------------------------------------
# Fast-path optimisation: pure text delta via _chunk_parser
# ---------------------------------------------------------------------------


def test_chunk_parser_pure_text_delta_returns_gchunk():
    """
    Pure text delta chunks must take the GChunk fast path so that no Pydantic
    ModelResponseStream / StreamingChoices / Delta objects are constructed.
    """
    chunk_data = {"contentBlockIndex": 0, "delta": {"text": "Hello, world!"}}
    decoder = AWSEventStreamDecoder(model="bedrock/anthropic.claude-3-sonnet-v1:0")
    result = decoder._chunk_parser(chunk_data)

    assert isinstance(result, dict), "Expected a GChunk (dict), got ModelResponseStream"
    assert result["text"] == "Hello, world!"
    assert result["is_finished"] is False
    assert result["finish_reason"] == ""
    assert result["usage"] is None


def test_chunk_parser_tool_use_delta_slow_path():
    """
    Tool-use delta chunks must NOT take the GChunk fast path because they carry
    a ChatCompletionToolCallChunk that only ModelResponseStream can convey.
    """
    chunk_data = {
        "contentBlockIndex": 1,
        "delta": {"toolUse": {"input": '{"key": "val"}'}},
    }
    decoder = AWSEventStreamDecoder(model="bedrock/anthropic.claude-3-sonnet-v1:0")
    decoder.tool_calls_index = 0  # simulate an active tool call
    result = decoder._chunk_parser(chunk_data)

    # Must NOT be a plain dict — must be a full ModelResponseStream
    assert hasattr(result, "choices"), "Expected ModelResponseStream for tool-use delta"


def test_chunk_parser_reasoning_delta_slow_path():
    """
    Reasoning-content deltas must NOT take the GChunk fast path because they
    carry thinking_blocks that only ModelResponseStream can convey.
    """
    chunk_data = {
        "contentBlockIndex": 0,
        "delta": {"reasoningContent": {"text": "Let me think..."}},
    }
    decoder = AWSEventStreamDecoder(model="bedrock/anthropic.claude-3-sonnet-v1:0")
    result = decoder._chunk_parser(chunk_data)

    assert hasattr(
        result, "choices"
    ), "Expected ModelResponseStream for reasoning delta"


def test_chunk_parser_trace_chunk_slow_path():
    """
    Chunks that contain a 'trace' field must NOT take the GChunk fast path
    because the trace data must be attached as provider_specific_fields.
    """
    chunk_data = {
        "contentBlockIndex": 0,
        "delta": {"text": "hi"},
        "trace": {"guardrail": {"inputAssessment": {}}},
    }
    decoder = AWSEventStreamDecoder(model="bedrock/anthropic.claude-3-sonnet-v1:0")
    result = decoder._chunk_parser(chunk_data)

    assert hasattr(result, "choices"), "Expected ModelResponseStream for trace chunk"


def test_chunk_parser_text_delta_with_padding_field_returns_gchunk():
    """
    Real Bedrock Converse streams include a 'p' padding field at the outer
    chunk level.  It must not affect the fast-path detection.
    """
    chunk_data = {
        "contentBlockIndex": 0,
        "delta": {"text": "padded chunk"},
        "p": "abcdefghijklmnopqrstuvwxyz",
    }
    decoder = AWSEventStreamDecoder(model="bedrock/anthropic.claude-3-sonnet-v1:0")
    result = decoder._chunk_parser(chunk_data)

    assert isinstance(result, dict), "Padding field should not disable the fast path"
    assert result["text"] == "padded chunk"


def test_chunk_parser_stop_reason_slow_path():
    """
    Stop-reason events have no 'delta' key and must not hit the fast path.
    """
    chunk_data = {"stopReason": "end_turn"}
    decoder = AWSEventStreamDecoder(model="bedrock/anthropic.claude-3-sonnet-v1:0")
    result = decoder._chunk_parser(chunk_data)

    # stop-reason chunks always go through converse_chunk_parser → ModelResponseStream
    assert hasattr(result, "choices"), "Expected ModelResponseStream for stop-reason"
    assert result.choices[0].finish_reason is not None


def test_chunk_parser_nova_invoke_text_delta_fast_path():
    """
    For Bedrock Invoke with Nova models the payload is wrapped under the key
    'contentBlockDelta'.  Pure text deltas must still hit the GChunk fast path.
    """
    chunk_data = {
        "contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {"text": "nova text"},
        }
    }
    decoder = AWSEventStreamDecoder(model="amazon.nova-pro-v1:0")
    result = decoder._chunk_parser(chunk_data)

    assert isinstance(result, dict), "Expected GChunk for nova invoke text delta"
    assert result["text"] == "nova text"
    assert result["is_finished"] is False


def test_chunk_parser_stream_text_parity():
    """
    Fast-path GChunk and slow-path ModelResponseStream must carry identical
    text content for the same pure text delta event.
    """
    chunk_data = {"contentBlockIndex": 0, "delta": {"text": "parity check"}}

    decoder_fast = AWSEventStreamDecoder(model="bedrock/test")
    fast_result = decoder_fast._chunk_parser(chunk_data)

    # Obtain the slow-path result by calling converse_chunk_parser directly
    decoder_slow = AWSEventStreamDecoder(model="bedrock/test")
    slow_result = decoder_slow.converse_chunk_parser(chunk_data)

    assert isinstance(fast_result, dict)
    assert fast_result["text"] == slow_result.choices[0].delta.content
