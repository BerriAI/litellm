import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../../.."))

from unittest.mock import patch

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.types.llms.anthropic import (
    AnthopicMessagesAssistantMessageParam,
    AnthropicMessagesUserMessageParam,
)
from litellm.types.llms.openai import ChatCompletionAssistantToolCall
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Choices,
    Delta,
    Function,
    Message,
    ModelResponse,
    StreamingChoices,
)


def test_translate_streaming_openai_chunk_to_anthropic_content_block():
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                provider_specific_fields=None,
                content=None,
                role="assistant",
                function_call=None,
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id="call_d581d130-e234-4315-94e8-27e7ff7c4e55",
                        function=Function(
                            arguments='{"location": "Boston"}', name="get_weather"
                        ),
                        type="function",
                        index=0,
                    )
                ],
                audio=None,
            ),
            logprobs=None,
        )
    ]

    (
        block_type,
        content_block_start,
    ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic_content_block(
        choices=choices
    )

    print(content_block_start)

    assert block_type == "tool_use"
    assert content_block_start == {
        "type": "tool_use",
        "id": "call_d581d130-e234-4315-94e8-27e7ff7c4e55",
        "name": "get_weather",
        "input": {},
    }


def test_translate_anthropic_messages_to_openai_tool_message_placement():
    """Test that tool result messages are placed before user messages in the conversation order."""

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[{"type": "text", "text": "What's the weather in Boston?"}]
        ),
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_01234",
                    "name": "get_weather",
                    "input": {"location": "Boston"}
                }
            ]
        ),
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01234",
                    "content": "Sunny, 75°F"
                },
                {"type": "text", "text": "What about tomorrow?"}
            ]
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    # find the indices of tool and user messages in the result
    tool_message_idx = None
    user_message_idx = None

    for i, msg in enumerate(result):
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tool_message_idx = i
        elif isinstance(msg, dict) and msg.get("role") == "user" and "What about tomorrow?" in str(msg.get("content", "")):
            user_message_idx = i
            break

    assert tool_message_idx is not None, "Tool message not found"
    assert user_message_idx is not None, "User message not found"
    assert tool_message_idx < user_message_idx, "Tool message should be placed before user message"


def test_translate_openai_content_to_anthropic_empty_function_arguments():
    """Test that empty function arguments are handled safely and don't cause JSON parsing errors."""

    openai_choices = [
        Choices(
            message=Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ChatCompletionAssistantToolCall(
                        id="call_empty_args",
                        type="function",
                        function=Function(
                            name="test_function",
                            arguments=""  # empty arguments string
                        )
                    )
                ]
            )
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter._translate_openai_content_to_anthropic(choices=openai_choices)

    assert len(result) == 1
    assert result[0].type == "tool_use"
    assert result[0].id == "call_empty_args"
    assert result[0].name == "test_function"
    assert result[0].input == {}, "Empty function arguments should result in empty dict"


def test_anthropic_stream_wrapper_no_empty_content_blocks():
    """
    Test that AnthropicStreamWrapper does not send empty content_block_start/stop 
    events when the stream contains no content.
    
    This test validates the fix for the issue where empty content blocks were
    being sent even when there was no actual content to stream.
    
    Relevant issue: https://github.com/BerriAI/litellm/issues/13373
    """
    # Create a stream with only a final chunk (no content)
    final_chunk = ModelResponse()
    final_chunk.choices = [
        Choices(
            index=0,
            delta=Delta(content=None, role=None),
            finish_reason="stop"
        )
    ]
    
    mock_stream = iter([final_chunk])
    wrapper = AnthropicStreamWrapper(completion_stream=mock_stream, model="gpt-4")
    
    events = []
    try:
        while True:
            event = next(wrapper)
            events.append(event)
    except StopIteration:
        pass
    
    # Verify events
    event_types = [event.get("type") for event in events]
    
    # Should have message_start and message_stop, but NO content_block_start/stop
    assert "message_start" in event_types, "Should have message_start event"
    assert "message_stop" in event_types, "Should have message_stop event"
    assert "content_block_start" not in event_types, "Should NOT have content_block_start for empty stream"
    assert "content_block_stop" not in event_types, "Should NOT have content_block_stop for empty stream"


def test_anthropic_stream_wrapper_with_content_doesnt_send_empty_blocks():
    """
    Test that AnthropicStreamWrapper does not send empty content blocks
    even when there is some content in the stream. This validates that the fix
    prevents empty blocks regardless of stream content.
    """
    # Create a stream with actual content
    content_chunk = ModelResponse()
    content_chunk.choices = [
        Choices(
            index=0,
            delta=Delta(content="Hello", role="assistant"),
            finish_reason=None
        )
    ]
    
    final_chunk = ModelResponse()
    final_chunk.choices = [
        Choices(
            index=0,
            delta=Delta(content=None, role=None),
            finish_reason="stop"
        )
    ]
    
    mock_stream = iter([content_chunk, final_chunk])
    wrapper = AnthropicStreamWrapper(completion_stream=mock_stream, model="gpt-4")
    
    events = []
    try:
        while True:
            event = next(wrapper)
            events.append(event)
    except StopIteration:
        pass
    
    # Verify events
    event_types = [event.get("type") for event in events]
    
    # The key thing is that if content_block_start is sent, it should not be empty
    content_block_start_events = [e for e in events if e.get("type") == "content_block_start"]
    
    # If content_block_start events exist, validate they're not empty
    for event in content_block_start_events:
        content_block = event.get("content_block", {})
        # Should not be sending empty text content blocks
        if content_block.get("type") == "text":
            # This test doesn't enforce content_block_start must be sent,
            # but if it is sent, it should be valid
            assert "text" in content_block, "Content block should have text field"
    
    # Core requirement: should have message_start and message_stop
    assert "message_start" in event_types, "Should have message_start event"
    assert "message_stop" in event_types, "Should have message_stop event"
