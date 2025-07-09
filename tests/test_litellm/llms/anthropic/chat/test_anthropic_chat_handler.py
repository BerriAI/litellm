import json
import os
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.anthropic.chat.handler import ModelResponseIterator
from litellm.types.llms.openai import ChatCompletionToolCallChunk, ChatCompletionToolCallFunctionChunk
from litellm.constants import RESPONSE_FORMAT_TOOL_NAME


def test_redacted_thinking_content_block_delta():
    chunk = {
        "type": "content_block_start",
        "index": 58,
        "content_block": {
            "type": "redacted_thinking",
            "data": "EuoBCoYBGAIiQJ/SxkPAgqxhKok29YrpJHRUJ0OT8ahCHKAwyhmRuUhtdmDX9+mn4gDzKNv3fVpQdB01zEPMzNY3QuTCd+1bdtEqQK6JuKHqdndbwpr81oVWb4wxd1GqF/7Jkw74IlQa27oobX+KuRkopr9Dllt/RDe7Se0sI1IkU7tJIAQCoP46OAwSDF51P09q67xhHlQ3ihoM2aOVlkghq/X0w8NlIjBMNvXYNbjhyrOcIg6kPFn2ed/KK7Cm5prYAtXCwkb4Wr5tUSoSHu9T5hKdJRbr6WsqEc7Lle7FULqMLZGkhqXyc3BA",
        },
    }
    model_response_iterator = ModelResponseIterator(
        streaming_response=MagicMock(), sync_stream=False, json_mode=False
    )
    model_response = model_response_iterator.chunk_parser(chunk=chunk)
    print(f"\n\nmodel_response: {model_response}\n\n")
    assert model_response.choices[0].delta.thinking_blocks is not None
    assert len(model_response.choices[0].delta.thinking_blocks) == 1
    print(
        f"\n\nmodel_response.choices[0].delta.thinking_blocks[0]: {model_response.choices[0].delta.thinking_blocks[0]}\n\n"
    )
    assert (
        model_response.choices[0].delta.thinking_blocks[0]["type"]
        == "redacted_thinking"
    )

    assert model_response.choices[0].delta.provider_specific_fields is not None
    assert "thinking_blocks" in model_response.choices[0].delta.provider_specific_fields


def test_handle_json_mode_chunk_response_format_tool():
    model_response_iterator = ModelResponseIterator(
        streaming_response=MagicMock(), sync_stream=True, json_mode=True
    )
    response_format_tool = ChatCompletionToolCallChunk(
        id="tool_123",
        type="function",
        function=ChatCompletionToolCallFunctionChunk(
            name=RESPONSE_FORMAT_TOOL_NAME,
            arguments='{"question": "What is the weather?", "answer": "It is sunny"}'
        ),
        index=0
    )
    
    text, tool_use = model_response_iterator._handle_json_mode_chunk("", response_format_tool)
    print(f"\n\nresponse_format_tool text: {text}\n\n")
    print(f"\n\nresponse_format_tool tool_use: {tool_use}\n\n")
    
    assert text == '{"question": "What is the weather?", "answer": "It is sunny"}'
    assert tool_use is None


def test_handle_json_mode_chunk_regular_tool():
    model_response_iterator = ModelResponseIterator(
        streaming_response=MagicMock(), sync_stream=True, json_mode=True
    )
    regular_tool = ChatCompletionToolCallChunk(
        id="tool_456",
        type="function",
        function=ChatCompletionToolCallFunctionChunk(
            name="get_weather",
            arguments='{"location": "San Francisco, CA"}'
        ),
        index=0
    )
    
    text, tool_use = model_response_iterator._handle_json_mode_chunk("", regular_tool)
    print(f"\n\nregular_tool text: {text}\n\n")
    print(f"\n\nregular_tool tool_use: {tool_use}\n\n")
    
    assert text == ""
    assert tool_use is not None
    assert tool_use["function"]["name"] == "get_weather"
