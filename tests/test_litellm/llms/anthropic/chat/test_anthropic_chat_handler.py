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


def test_partial_json_chunk_accumulation():
    """
    Test that partial JSON chunks are accumulated correctly.

    This tests the fix for https://github.com/BerriAI/litellm/issues/17473
    where network fragmentation can cause SSE data to arrive in partial chunks.
    """
    iterator = ModelResponseIterator(
        streaming_response=MagicMock(), sync_stream=True, json_mode=False
    )

    # Simulate a complete JSON chunk being split into two parts
    partial_chunk_1 = '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel'
    partial_chunk_2 = 'lo"}}'

    # First partial chunk should return None (still accumulating)
    result1 = iterator._parse_sse_data(f"data:{partial_chunk_1}")
    assert result1 is None, "First partial chunk should return None while accumulating"
    assert iterator.chunk_type == "accumulated_json", "Should switch to accumulated_json mode"
    assert iterator.accumulated_json == partial_chunk_1, "Should have accumulated first part"

    # Second partial chunk should complete the JSON and return a parsed result
    result2 = iterator._parse_sse_data(f"data:{partial_chunk_2}")
    assert result2 is not None, "Second chunk should return parsed result"
    assert iterator.accumulated_json == "", "Buffer should be cleared after successful parse"
    assert result2.choices[0].delta.content == "Hello", f"Expected 'Hello', got '{result2.choices[0].delta.content}'"


def test_complete_json_chunk_no_accumulation():
    """
    Test that complete JSON chunks are parsed immediately without accumulation.
    """
    iterator = ModelResponseIterator(
        streaming_response=MagicMock(), sync_stream=True, json_mode=False
    )

    complete_chunk = '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}'

    result = iterator._parse_sse_data(f"data:{complete_chunk}")
    assert result is not None, "Complete chunk should return parsed result immediately"
    assert iterator.chunk_type == "valid_json", "Should remain in valid_json mode"
    assert iterator.accumulated_json == "", "Buffer should remain empty"
    assert result.choices[0].delta.content == "Hello", f"Expected 'Hello', got '{result.choices[0].delta.content}'"


def test_multiple_partial_chunks_accumulation():
    """
    Test that multiple partial chunks can be accumulated across several iterations.
    """
    iterator = ModelResponseIterator(
        streaming_response=MagicMock(), sync_stream=True, json_mode=False
    )

    # Split a JSON chunk into three parts
    part1 = '{"type":"content_block_del'
    part2 = 'ta","index":0,"delta":{"type":"text_del'
    part3 = 'ta","text":"Hello"}}'

    result1 = iterator._parse_sse_data(f"data:{part1}")
    assert result1 is None
    assert iterator.accumulated_json == part1

    result2 = iterator._parse_sse_data(f"data:{part2}")
    assert result2 is None
    assert iterator.accumulated_json == part1 + part2

    result3 = iterator._parse_sse_data(f"data:{part3}")
    assert result3 is not None
    assert iterator.accumulated_json == ""
    assert result3.choices[0].delta.content == "Hello"


def test_web_search_tool_result_no_extra_tool_calls():
    """
    Test that web_search_tool_result blocks don't emit tool call chunks.

    This tests the fix for https://github.com/BerriAI/litellm/issues/17254
    where streaming with Anthropic web search was adding trailing {} to tool call arguments.

    The issue was that web_search_tool_result blocks have input_json_delta events with {}
    that were incorrectly being converted to tool calls.
    """
    iterator = ModelResponseIterator(
        streaming_response=MagicMock(), sync_stream=True, json_mode=False
    )

    # Simulate the streaming sequence:
    # 1. server_tool_use block starts (web_search)
    # 2. input_json_delta with the query
    # 3. content_block_stop
    # 4. web_search_tool_result block starts
    # 5. input_json_delta with {} (this should NOT emit a tool call)
    # 6. content_block_stop

    chunks = [
        # 1. server_tool_use block starts
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_01ABC123",
                "name": "web_search",
            },
        },
        # 2. input_json_delta with the query
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"query": "test"}'},
        },
        # 3. content_block_stop for server_tool_use
        {"type": "content_block_stop", "index": 0},
        # 4. web_search_tool_result block starts
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_01ABC123",
                "content": [],
            },
        },
        # 5. input_json_delta with {} - this should NOT emit a tool call
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "{}"},
        },
        # 6. content_block_stop for web_search_tool_result
        {"type": "content_block_stop", "index": 1},
        # 7. Another web_search_tool_result with {} - also should NOT emit
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_01ABC123",
                "content": [],
            },
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": "{}"},
        },
        {"type": "content_block_stop", "index": 2},
    ]

    tool_calls_emitted = []
    for chunk in chunks:
        parsed = iterator.chunk_parser(chunk)
        if parsed.choices and parsed.choices[0].delta.tool_calls:
            for tc in parsed.choices[0].delta.tool_calls:
                tool_calls_emitted.append(tc)

    # Should have exactly 2 tool calls:
    # 1. From content_block_start (server_tool_use) with id and name
    # 2. From content_block_delta with the actual query
    assert len(tool_calls_emitted) == 2, f"Expected 2 tool calls, got {len(tool_calls_emitted)}"

    # First tool call should have the id and name
    assert tool_calls_emitted[0]["id"] == "srvtoolu_01ABC123"
    assert tool_calls_emitted[0]["function"]["name"] == "web_search"

    # Second tool call should have the query arguments
    assert tool_calls_emitted[1]["function"]["arguments"] == '{"query": "test"}'

    # The {} chunks from web_search_tool_result should NOT have been emitted as tool calls


def test_current_content_block_type_tracking():
    """
    Test that current_content_block_type is properly tracked and reset.
    """
    iterator = ModelResponseIterator(
        streaming_response=MagicMock(), sync_stream=True, json_mode=False
    )

    # Initially should be None
    assert iterator.current_content_block_type is None

    # After server_tool_use block start
    chunk1 = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "server_tool_use",
            "id": "srvtoolu_01ABC",
            "name": "web_search",
        },
    }
    iterator.chunk_parser(chunk1)
    assert iterator.current_content_block_type == "server_tool_use"

    # After content_block_stop
    chunk2 = {"type": "content_block_stop", "index": 0}
    iterator.chunk_parser(chunk2)
    assert iterator.current_content_block_type is None

    # After web_search_tool_result block start
    chunk3 = {
        "type": "content_block_start",
        "index": 1,
        "content_block": {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_01ABC",
            "content": [],
        },
    }
    iterator.chunk_parser(chunk3)
    assert iterator.current_content_block_type == "web_search_tool_result"

    # After content_block_stop
    chunk4 = {"type": "content_block_stop", "index": 1}
    iterator.chunk_parser(chunk4)
    assert iterator.current_content_block_type is None


def test_web_search_tool_result_captured_in_provider_specific_fields():
    """
    Test that web_search_tool_result content is captured in provider_specific_fields.

    This tests the fix for https://github.com/BerriAI/litellm/issues/17737
    where streaming with Anthropic web search wasn't capturing web_search_tool_result
    blocks, causing multi-turn conversations to fail.

    The web_search_tool_result content comes ALL AT ONCE in content_block_start,
    not in deltas, so we need to capture it there.
    """
    iterator = ModelResponseIterator(
        streaming_response=MagicMock(), sync_stream=True, json_mode=False
    )

    # Simulate the streaming sequence with web_search_tool_result
    chunks = [
        # 1. message_start
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
        # 2. server_tool_use block starts (web_search)
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_01ABC123",
                "name": "web_search",
            },
        },
        # 3. input_json_delta with the query
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"query": "otter facts"}'},
        },
        # 4. content_block_stop for server_tool_use
        {"type": "content_block_stop", "index": 0},
        # 5. web_search_tool_result block starts - THIS IS WHERE THE RESULTS ARE
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_01ABC123",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://example.com/otters",
                        "title": "Fun Otter Facts",
                        "encrypted_content": "abc123encrypted",
                    },
                    {
                        "type": "web_search_result",
                        "url": "https://example.com/otters2",
                        "title": "More Otter Facts",
                        "encrypted_content": "def456encrypted",
                    },
                ],
            },
        },
        # 6. content_block_stop for web_search_tool_result
        {"type": "content_block_stop", "index": 1},
    ]

    web_search_results = None
    for chunk in chunks:
        parsed = iterator.chunk_parser(chunk)
        if (
            parsed.choices
            and parsed.choices[0].delta.provider_specific_fields
            and "web_search_results" in parsed.choices[0].delta.provider_specific_fields
        ):
            web_search_results = parsed.choices[0].delta.provider_specific_fields[
                "web_search_results"
            ]

    # Verify web_search_results was captured
    assert web_search_results is not None, "web_search_results should be captured"
    assert len(web_search_results) == 1, "Should have 1 web_search_tool_result block"
    assert (
        web_search_results[0]["type"] == "web_search_tool_result"
    ), "Block type should be web_search_tool_result"
    assert (
        web_search_results[0]["tool_use_id"] == "srvtoolu_01ABC123"
    ), "tool_use_id should match"
    assert len(web_search_results[0]["content"]) == 2, "Should have 2 search results"
    assert (
        web_search_results[0]["content"][0]["title"] == "Fun Otter Facts"
    ), "First result title should match"
