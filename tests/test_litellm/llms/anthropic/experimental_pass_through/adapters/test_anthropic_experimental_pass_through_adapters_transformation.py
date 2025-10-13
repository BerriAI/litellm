import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../../.."))

from unittest.mock import patch

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


def test_translate_streaming_openai_chunk_to_anthropic_thinking_content_block():
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                reasoning_content="I need to summar",
                thinking_blocks=[
                    {
                        "type": "thinking",
                        "thinking": "I need to summar",
                        "signature": None,
                    }
                ],
                provider_specific_fields={
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I need to summar",
                            "signature": None,
                        }
                    ]
                },
                content="",
                role="assistant",
                function_call=None,
                tool_calls=None,
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

    assert block_type == "thinking"
    assert content_block_start == {
        "type": "thinking",
        "thinking": "I need to summar",
        "signature": "",
    }


def test_translate_streaming_openai_chunk_to_anthropic_thinking_signature_block():
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                reasoning_content="",
                thinking_blocks=[
                    {
                        "type": "thinking",
                        "thinking": None,
                        "signature": "sigsig",
                    }
                ],
                provider_specific_fields={
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": None,
                            "signature": "sigsig",
                        }
                    ]
                },
                content="",
                role="assistant",
                function_call=None,
                tool_calls=None,
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

    assert block_type == "thinking"
    assert content_block_start == {
        "type": "thinking",
        "thinking": "",
        "signature": "sigsig",
    }


def test_translate_streaming_openai_chunk_to_anthropic_raises_when_thinking_and_signature_content_block():
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                reasoning_content="",
                thinking_blocks=[
                    {
                        "type": "thinking",
                        "thinking": "I need to summar",
                        "signature": "sigsig",
                    }
                ],
                provider_specific_fields={
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I need to summar",
                            "signature": "sigsig",
                        }
                    ]
                },
                content="",
                role="assistant",
                function_call=None,
                tool_calls=None,
                audio=None,
            ),
            logprobs=None,
        )
    ]

    with pytest.raises(ValueError):
        LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic_content_block(
            choices=choices
        )


def test_translate_anthropic_messages_to_openai_thinking_blocks():
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
                    "type": "thinking",
                    "thinking": "I will call the get_weather tool.",
                    "signature": "sigsig"
                },
                {
                    "type": "redacted_thinking",
                    "data": "REDACTED",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01234",
                    "name": "get_weather",
                    "input": {"location": "Boston"} 
                }
            ]
        ),
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    assert len(result) == 2
    assert result[1]["role"] == "assistant"
    assert "thinking_blocks" in result[1]
    assert len(result[1]["thinking_blocks"]) == 2
    assert result[1]["thinking_blocks"][0]["type"] == "thinking"
    assert result[1]["thinking_blocks"][0]["thinking"] == "I will call the get_weather tool."
    assert result[1]["thinking_blocks"][0]["signature"] == "sigsig"
    assert result[1]["thinking_blocks"][1]["type"] == "redacted_thinking"
    assert result[1]["thinking_blocks"][1]["data"] == "REDACTED"
    assert "tool_calls" in result[1]
    assert len(result[1]["tool_calls"]) == 1
    assert result[1]["tool_calls"][0]["id"] == "toolu_01234"


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



def test_translate_streaming_openai_chunk_to_anthropic_with_partial_json():
    """Test that partial tool arguments are correctly handled as input_json_delta."""
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=1,
            delta=Delta(
                provider_specific_fields=None,
                content='',
                role='assistant',
                function_call=None,
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id=None,
                        function=Function(arguments=': "San ', name=None),
                        type='function',
                        index=0
                    )
                ],
                audio=None,
            ),
            logprobs=None,
        )
    ]

    (
        type_of_content,
        content_block_delta,
    ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic(
        choices=choices
    )

    print("Type of content:", type_of_content)
    print("Content block delta:", content_block_delta)

    assert type_of_content == "input_json_delta"
    assert content_block_delta["type"] == "input_json_delta" 
    assert content_block_delta["partial_json"] == ': "San '



def test_translate_openai_content_to_anthropic_thinking_and_redacted_thinking():
    openai_choices = [
        Choices(
            message=Message(
                role="assistant",
                content=None,
                thinking_blocks=[
                    {
                        "type": "thinking",
                        "thinking": "I need to summar",
                        "signature": "sigsig",
                    },
                    {
                        "type": "redacted_thinking",
                        "data": "REDACTED"
                    }
                ]
            )
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter._translate_openai_content_to_anthropic(choices=openai_choices)

    assert len(result) == 2
    assert result[0].type == "thinking"
    assert result[0].thinking == "I need to summar"
    assert result[0].signature == "sigsig"
    assert result[1].type == "redacted_thinking"
    assert result[1].data == "REDACTED"


def test_translate_streaming_openai_chunk_to_anthropic_with_thinking():
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                reasoning_content="I need to summar",
                thinking_blocks=[
                    {
                        "type": "thinking",
                        "thinking": "I need to summar",
                        "signature": None,
                    }
                ],
                provider_specific_fields={
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I need to summar",
                            "signature": None,
                        }
                    ]
                },
                content="",
                role="assistant",
                function_call=None,
                tool_calls=None,
                audio=None,
            ),
            logprobs=None,
        )
    ]

    (
        type_of_content,
        content_block_delta,
    ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic(
        choices=choices
    )

    assert type_of_content == "thinking_delta"
    assert content_block_delta["type"] == "thinking_delta" 
    assert content_block_delta["thinking"] == "I need to summar"


def test_translate_streaming_openai_chunk_to_anthropic_with_thinking():
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                reasoning_content="",
                thinking_blocks=[
                    {
                        "type": "thinking",
                        "thinking": None,
                        "signature": "sigsig",
                    }
                ],
                provider_specific_fields={
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": None,
                            "signature": "sigsig",
                        }
                    ]
                },
                content="",
                role="assistant",
                function_call=None,
                tool_calls=None,
                audio=None,
            ),
            logprobs=None,
        )
    ]

    (
        type_of_content,
        content_block_delta,
    ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic(
        choices=choices
    )

    assert type_of_content == "signature_delta"
    assert content_block_delta["type"] == "signature_delta" 
    assert content_block_delta["signature"] == "sigsig"


def test_translate_streaming_openai_chunk_to_anthropic_raises_when_thinking_and_signature():
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                reasoning_content="",
                thinking_blocks=[
                    {
                        "type": "thinking",
                        "thinking": "I need to summar",
                        "signature": "sigsig",
                    }
                ],
                provider_specific_fields={
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I need to summar",
                            "signature": "sigsig",
                        }
                    ]
                },
                content="",
                role="assistant",
                function_call=None,
                tool_calls=None,
                audio=None,
            ),
            logprobs=None,
        )
    ]

    with pytest.raises(ValueError):
        LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic(
            choices=choices
        )
