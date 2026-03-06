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
