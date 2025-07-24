import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../../.."))

from unittest.mock import patch

from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.types.llms.anthropic import AnthropicMessagesUserMessageParam, AnthopicMessagesAssistantMessageParam
from litellm.types.llms.openai import ChatCompletionAssistantToolCall
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    StreamingChoices,
    Choices,
    Message,
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
