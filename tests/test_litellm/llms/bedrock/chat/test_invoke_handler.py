import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

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
