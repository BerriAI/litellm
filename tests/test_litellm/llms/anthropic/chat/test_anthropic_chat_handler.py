from unittest.mock import MagicMock

from litellm.llms.anthropic.chat.handler import ModelResponseIterator
from litellm.types.llms.openai import (
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
)
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
            arguments='{"question": "What is the weather?", "answer": "It is sunny"}',
        ),
        index=0,
    )

    text, tool_use = model_response_iterator._handle_json_mode_chunk(
        "", response_format_tool
    )
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
            name="get_weather", arguments='{"location": "San Francisco, CA"}'
        ),
        index=0,
    )

    text, tool_use = model_response_iterator._handle_json_mode_chunk("", regular_tool)
    print(f"\n\nregular_tool text: {text}\n\n")
    print(f"\n\nregular_tool tool_use: {tool_use}\n\n")

    assert text == ""
    assert tool_use is not None
    assert tool_use["function"]["name"] == "get_weather"


def test_handle_json_mode_chunk_streaming_response_format_tool():
    model_response_iterator = ModelResponseIterator(
        streaming_response=MagicMock(), sync_stream=True, json_mode=True
    )

    # First chunk: response_format tool with id and name, but no arguments
    first_chunk = ChatCompletionToolCallChunk(
        id="tool_123",
        type="function",
        function=ChatCompletionToolCallFunctionChunk(
            name=RESPONSE_FORMAT_TOOL_NAME, arguments=""
        ),
        index=0,
    )

    # Second chunk: continuation with arguments delta (no id)
    second_chunk = ChatCompletionToolCallChunk(
        id=None,
        type="function",
        function=ChatCompletionToolCallFunctionChunk(
            name=None, arguments='{"question": "What is the weather?"'
        ),
        index=0,
    )

    # Third chunk: more arguments delta (no id)
    third_chunk = ChatCompletionToolCallChunk(
        id=None,
        type="function",
        function=ChatCompletionToolCallFunctionChunk(
            name=None, arguments=', "answer": "It is sunny"}'
        ),
        index=0,
    )

    # Process first chunk - should set tracking flag but not convert yet (no args)
    text1, tool_use1 = model_response_iterator._handle_json_mode_chunk("", first_chunk)
    print(f"\n\nfirst_chunk text: {text1}\n\n")
    print(f"\n\nfirst_chunk tool_use: {tool_use1}\n\n")

    # Process second chunk - should convert arguments to text
    text2, tool_use2 = model_response_iterator._handle_json_mode_chunk("", second_chunk)
    print(f"\n\nsecond_chunk text: {text2}\n\n")
    print(f"\n\nsecond_chunk tool_use: {tool_use2}\n\n")

    # Process third chunk - should convert arguments to text
    text3, tool_use3 = model_response_iterator._handle_json_mode_chunk("", third_chunk)
    print(f"\n\nthird_chunk text: {text3}\n\n")
    print(f"\n\nthird_chunk tool_use: {tool_use3}\n\n")

    # Verify response_format tool chunks are converted to content
    assert text1 == ""  # First chunk has no arguments
    assert tool_use1 is None  # Tool call suppressed

    assert text2 == '{"question": "What is the weather?"'  # Second chunk arguments
    assert tool_use2 is None  # Tool call suppressed

    assert text3 == ', "answer": "It is sunny"}'  # Third chunk arguments
    assert tool_use3 is None  # Tool call suppressed


def test_handle_json_mode_chunk_streaming_regular_tool():
    model_response_iterator = ModelResponseIterator(
        streaming_response=MagicMock(), sync_stream=True, json_mode=True
    )

    # First chunk: regular tool with id and name, but no arguments
    first_chunk = ChatCompletionToolCallChunk(
        id="tool_456",
        type="function",
        function=ChatCompletionToolCallFunctionChunk(name="get_weather", arguments=""),
        index=0,
    )

    # Second chunk: continuation with arguments delta (no id)
    second_chunk = ChatCompletionToolCallChunk(
        id=None,
        type="function",
        function=ChatCompletionToolCallFunctionChunk(
            name=None, arguments='{"location": "San Francisco, CA"}'
        ),
        index=0,
    )

    # Process first chunk - should pass through as regular tool
    text1, tool_use1 = model_response_iterator._handle_json_mode_chunk("", first_chunk)
    print(f"\n\nregular first_chunk text: {text1}\n\n")
    print(f"\n\nregular first_chunk tool_use: {tool_use1}\n\n")

    # Process second chunk - should pass through as regular tool
    text2, tool_use2 = model_response_iterator._handle_json_mode_chunk("", second_chunk)
    print(f"\n\nregular second_chunk text: {text2}\n\n")
    print(f"\n\nregular second_chunk tool_use: {tool_use2}\n\n")

    # Verify regular tool chunks are passed through unchanged
    assert text1 == ""  # Original text unchanged
    assert tool_use1 is not None  # Tool call preserved
    assert tool_use1["function"]["name"] == "get_weather"

    assert text2 == ""  # Original text unchanged
    assert tool_use2 is not None  # Tool call preserved
    assert tool_use2["function"]["arguments"] == '{"location": "San Francisco, CA"}'


def test_response_format_tool_finish_reason():
    model_response_iterator = ModelResponseIterator(
        streaming_response=MagicMock(), sync_stream=True, json_mode=True
    )

    # First chunk: response_format tool
    response_format_tool = ChatCompletionToolCallChunk(
        id="tool_123",
        type="function",
        function=ChatCompletionToolCallFunctionChunk(
            name=RESPONSE_FORMAT_TOOL_NAME, arguments='{"answer": "test"}'
        ),
        index=0,
    )

    # Process the tool call (should set converted_response_format_tool flag)
    text, tool_use = model_response_iterator._handle_json_mode_chunk(
        "", response_format_tool
    )
    print(
        f"\n\nconverted_response_format_tool flag: {model_response_iterator.converted_response_format_tool}\n\n"
    )

    # Simulate message_delta chunk with tool_use stop_reason
    message_delta_chunk = {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": 10},
    }

    # Process the message_delta chunk
    model_response = model_response_iterator.chunk_parser(message_delta_chunk)
    print(f"\n\nfinish_reason: {model_response.choices[0].finish_reason}\n\n")

    # Verify that finish_reason is overridden to "stop" for response_format tools
    assert model_response_iterator.converted_response_format_tool is True
    assert model_response.choices[0].finish_reason == "stop"


def test_regular_tool_finish_reason():
    model_response_iterator = ModelResponseIterator(
        streaming_response=MagicMock(), sync_stream=True, json_mode=True
    )

    # First chunk: regular tool (not response_format)
    regular_tool = ChatCompletionToolCallChunk(
        id="tool_456",
        type="function",
        function=ChatCompletionToolCallFunctionChunk(
            name="get_weather", arguments='{"location": "San Francisco, CA"}'
        ),
        index=0,
    )

    # Process the tool call (should NOT set converted_response_format_tool flag)
    text, tool_use = model_response_iterator._handle_json_mode_chunk("", regular_tool)
    print(
        f"\n\nconverted_response_format_tool flag: {model_response_iterator.converted_response_format_tool}\n\n"
    )

    # Simulate message_delta chunk with tool_use stop_reason
    message_delta_chunk = {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": 10},
    }

    # Process the message_delta chunk
    model_response = model_response_iterator.chunk_parser(message_delta_chunk)
    print(f"\n\nfinish_reason: {model_response.choices[0].finish_reason}\n\n")

    # Verify that finish_reason remains "tool_calls" for regular tools
    assert model_response_iterator.converted_response_format_tool is False
    assert model_response.choices[0].finish_reason == "tool_calls"


def test_text_only_streaming_has_index_zero():
    """Test that text-only streaming responses have choice index=0"""
    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": " world"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 2},
        },
    ]

    iterator = ModelResponseIterator(None, sync_stream=True)

    # Check all chunks have choice index=0
    for chunk in chunks:
        parsed = iterator.chunk_parser(chunk)
        if parsed.choices:
            assert (
                parsed.choices[0].index == 0
            ), f"Expected index=0, got {parsed.choices[0].index}"


def test_text_and_tool_streaming_has_index_zero():
    """Test that mixed text and tool streaming responses have choice index=0"""
    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        },
        # Reasoning content at index 0
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "I need to search..."},
        },
        {"type": "content_block_stop", "index": 0},
        # Regular content at index 1
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "Let me help you"},
        },
        {"type": "content_block_stop", "index": 1},
        # Tool call at index 2
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "tool_use",
                "id": "tool_123",
                "name": "search",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"query"'},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": ': "test"}'},
        },
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 10},
        },
    ]

    iterator = ModelResponseIterator(None, sync_stream=True)

    # Check all chunks have choice index=0 despite different Anthropic indices
    for chunk in chunks:
        parsed = iterator.chunk_parser(chunk)
        if parsed.choices:
            assert (
                parsed.choices[0].index == 0
            ), f"Expected index=0 for chunk type {chunk.get('type')}, got {parsed.choices[0].index}"


def test_multiple_tools_streaming_has_index_zero():
    """Test that multiple tool calls all have choice index=0"""
    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        },
        # First tool at index 0
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "tool_1",
                "name": "search",
                "input": {},
            },
        },
        {"type": "content_block_stop", "index": 0},
        # Second tool at index 1
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "tool_2",
                "name": "get",
                "input": {},
            },
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 5},
        },
    ]

    iterator = ModelResponseIterator(None, sync_stream=True)

    # All tool chunks should have choice index=0
    for chunk in chunks:
        parsed = iterator.chunk_parser(chunk)
        if parsed.choices:
            assert (
                parsed.choices[0].index == 0
            ), f"Expected index=0, got {parsed.choices[0].index}"


def test_streaming_chunks_have_stable_ids():
    iterator = ModelResponseIterator(
        streaming_response=MagicMock(), sync_stream=False, json_mode=False
    )
    first_chunk = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hello"},
    }
    second_chunk = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " world"},
    }

    response_one = iterator.chunk_parser(chunk=first_chunk)
    response_two = iterator.chunk_parser(chunk=second_chunk)

    assert response_one.id == response_two.id == iterator.response_id
