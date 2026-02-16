import os
import sys
from typing import Any, cast

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))


from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    OPENAI_MAX_TOOL_NAME_LENGTH,
    LiteLLMAnthropicMessagesAdapter,
    create_tool_name_mapping,
    truncate_tool_name,
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
    Usage,
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
            content=[{"type": "text", "text": "What's the weather in Boston?"}],
        ),
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "thinking",
                    "thinking": "I will call the get_weather tool.",
                    "signature": "sigsig",
                },
                {
                    "type": "redacted_thinking",
                    "data": "REDACTED",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01234",
                    "name": "get_weather",
                    "input": {"location": "Boston"},
                },
            ],
        ),
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    assert len(result) == 2
    assert result[1]["role"] == "assistant"
    assert "thinking_blocks" in result[1]
    assert len(result[1]["thinking_blocks"]) == 2
    assert result[1]["thinking_blocks"][0]["type"] == "thinking"
    assert (
        result[1]["thinking_blocks"][0]["thinking"]
        == "I will call the get_weather tool."
    )
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
            content=[{"type": "text", "text": "What's the weather in Boston?"}],
        ),
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_01234",
                    "name": "get_weather",
                    "input": {"location": "Boston"},
                }
            ],
        ),
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01234",
                    "content": "Sunny, 75°F",
                },
                {"type": "text", "text": "What about tomorrow?"},
            ],
        ),
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    # find the indices of tool and user messages in the result
    tool_message_idx = None
    user_message_idx = None

    for i, msg in enumerate(result):
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tool_message_idx = i
        elif (
            isinstance(msg, dict)
            and msg.get("role") == "user"
            and "What about tomorrow?" in str(msg.get("content", ""))
        ):
            user_message_idx = i
            break

    assert tool_message_idx is not None, "Tool message not found"
    assert user_message_idx is not None, "User message not found"
    assert (
        tool_message_idx < user_message_idx
    ), "Tool message should be placed before user message"


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
                            name="test_function", arguments=""  # empty arguments string
                        ),
                    )
                ],
            )
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter._translate_openai_content_to_anthropic(choices=openai_choices)

    assert len(result) == 1
    assert result[0]["type"] == "tool_use"
    assert result[0]["id"] == "call_empty_args"
    assert result[0]["name"] == "test_function"
    assert result[0]["input"] == {}, "Empty function arguments should result in empty dict"


def test_translate_openai_content_to_anthropic_text_and_tool_calls():
    """Ensure content blocks contain both the assistant text + tool call data."""
    openai_choices = [
        Choices(
            message=Message(
                role="assistant",
                content="Calling get_weather now.",
                tool_calls=[
                    ChatCompletionAssistantToolCall(
                        id="call_weather",
                        type="function",
                        function=Function(
                            name="get_weather",
                            arguments='{"location": "Boston"}',
                        ),
                    )
                ],
            )
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter._translate_openai_content_to_anthropic(choices=openai_choices)

    assert len(result) == 2
    assert result[0]["type"] == "text"
    assert result[0]["text"] == "Calling get_weather now."
    assert result[1]["type"] == "tool_use"
    assert result[1]["id"] == "call_weather"
    assert result[1]["name"] == "get_weather"
    assert result[1]["input"] == {"location": "Boston"}


def test_translate_openai_response_to_anthropic_text_and_tool_calls():
    """`translate_openai_response_to_anthropic` should surface assistant text even when tools fire."""
    openai_response = ModelResponse(
        id="resp_text_tool",
        model="gpt-4o-mini",
        choices=[
            Choices(
                finish_reason="tool_calls",
                message=Message(
                    role="assistant",
                    content="Let me grab the current weather.",
                    tool_calls=[
                        ChatCompletionAssistantToolCall(
                            id="call_tool_combo",
                            type="function",
                            function=Function(
                                name="get_weather", arguments='{"location": "Paris"}'
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=Usage(prompt_tokens=5, completion_tokens=2),
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    anthropic_response = adapter.translate_openai_response_to_anthropic(
        response=openai_response
    )

    anthropic_content = anthropic_response.get("content")
    assert anthropic_content is not None
    assert len(anthropic_content) == 2
    assert anthropic_content[0]["type"] == "text"
    assert anthropic_content[0]["text"] == "Let me grab the current weather."
    assert anthropic_content[1]["type"] == "tool_use"
    assert anthropic_content[1]["id"] == "call_tool_combo"
    assert anthropic_content[1]["input"] == {"location": "Paris"}
    assert anthropic_response.get("stop_reason") == "tool_use"


def test_translate_streaming_openai_chunk_to_anthropic_with_partial_json():
    """Test that partial tool arguments are correctly handled as input_json_delta."""
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=1,
            delta=Delta(
                provider_specific_fields=None,
                content="",
                role="assistant",
                function_call=None,
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id=None,
                        function=Function(arguments=': "San ', name=None),
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
                    {"type": "redacted_thinking", "data": "REDACTED"},
                ],
            )
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter._translate_openai_content_to_anthropic(choices=openai_choices)

    assert len(result) == 2
    assert result[0]["type"] == "thinking"
    assert result[0]["thinking"] == "I need to summar"
    assert result[0]["signature"] == "sigsig"
    assert result[1]["type"] == "redacted_thinking"
    assert result[1]["data"] == "REDACTED"


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


def test_translate_anthropic_messages_to_openai_user_message_with_base64_image():
    """Test that base64 images in user messages are correctly translated to OpenAI format."""

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    },
                },
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], list)
    assert len(result[0]["content"]) == 2

    # Check text content
    assert result[0]["content"][0]["type"] == "text"
    assert result[0]["content"][0]["text"] == "What's in this image?"

    # Check image content
    assert result[0]["content"][1]["type"] == "image_url"
    assert "image_url" in result[0]["content"][1]
    assert result[0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        in result[0]["content"][1]["image_url"]["url"]
    )


def test_translate_anthropic_messages_to_openai_user_message_with_url_image():
    """Test that URL-based images in user messages are correctly translated to OpenAI format."""

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {"type": "text", "text": "Describe this forest path"},
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/forest.jpg"},
                },
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], list)
    assert len(result[0]["content"]) == 2

    # Check text content
    assert result[0]["content"][0]["type"] == "text"
    assert result[0]["content"][0]["text"] == "Describe this forest path"

    # Check image content
    assert result[0]["content"][1]["type"] == "image_url"
    assert "image_url" in result[0]["content"][1]
    assert (
        result[0]["content"][1]["image_url"]["url"] == "https://example.com/forest.jpg"
    )


def test_translate_anthropic_messages_to_openai_tool_result_with_base64_image():
    """Test that base64 images in tool results are correctly translated to OpenAI format."""

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user", content=[{"type": "text", "text": "Take a screenshot"}]
        ),
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_01A09q90qw90lq917835lq9",
                    "name": "get_screenshot",
                    "input": {"area": "desktop"},
                }
            ],
        ),
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/wA/==",
                            },
                        }
                    ],
                }
            ],
        ),
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    # Find the tool message in the result
    tool_message = None
    for msg in result:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tool_message = msg
            break

    assert tool_message is not None, "Tool message not found in result"
    # Tool messages in OpenAI format have string content (data URL), not list
    assert isinstance(tool_message["content"], str)
    assert tool_message["content"].startswith("data:image/jpeg;base64,")
    assert "/9j/4AAQSkZJRgABAQAAAQABAAD" in tool_message["content"]


def test_translate_anthropic_messages_to_openai_tool_result_with_url_image():
    """Test that URL-based images in tool results are correctly translated to OpenAI format."""

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[{"type": "text", "text": "Take a screenshot of the forest"}],
        ),
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_01A09q90qw90lq917835lq9",
                    "name": "get_screenshot",
                    "input": {"area": "forest_path"},
                }
            ],
        ),
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://i0.wp.com/picjumbo.com/wp-content/uploads/amazing-stone-path-in-forest-free-image.jpg",
                            },
                        }
                    ],
                }
            ],
        ),
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    # Find the tool message in the result
    tool_message = None
    for msg in result:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tool_message = msg
            break

    assert tool_message is not None, "Tool message not found in result"
    # Tool messages in OpenAI format have string content (URL), not list
    assert isinstance(tool_message["content"], str)
    assert (
        tool_message["content"]
        == "https://i0.wp.com/picjumbo.com/wp-content/uploads/amazing-stone-path-in-forest-free-image.jpg"
    )


def test_translate_anthropic_messages_to_openai_mixed_content_with_image():
    """Test that messages with mixed text and image content are correctly translated."""

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {"type": "text", "text": "Here are two images:"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    },
                },
                {"type": "text", "text": "and this one:"},
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/image2.jpg"},
                },
                {"type": "text", "text": "What's the difference?"},
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], list)
    assert len(result[0]["content"]) == 5

    # Check text content
    assert result[0]["content"][0]["type"] == "text"
    assert result[0]["content"][0]["text"] == "Here are two images:"

    # Check first image (base64)
    assert result[0]["content"][1]["type"] == "image_url"
    assert result[0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )

    # Check middle text
    assert result[0]["content"][2]["type"] == "text"
    assert result[0]["content"][2]["text"] == "and this one:"

    # Check second image (URL)
    assert result[0]["content"][3]["type"] == "image_url"
    assert (
        result[0]["content"][3]["image_url"]["url"] == "https://example.com/image2.jpg"
    )

    # Check final text
    assert result[0]["content"][4]["type"] == "text"
    assert result[0]["content"][4]["text"] == "What's the difference?"


def test_translate_anthropic_messages_to_openai_tool_use_with_signature():
    """Test that thought signatures from tool_use blocks are correctly extracted and placed in provider_specific_fields."""

    test_signature = "EpYECpMEAdHtim9iBECdK1l5uVIIXoZZmq+PUBH9nz3Q6EMeIdEqWwVb5GlxSNtxuSkFoseFco5U4zxN/lacJxD2WUjFvEyL2GOkbPgXFeCcgNBMEYVRg7UAr45KGeWJJmJMoheLHezKawI1L94vi2PsB9TDpWv4vyAx1vKG2PByiVmWWtd0rondsdbENNp2Rrz3ol1zha+XhOtyhTCdSWce8GVD/zElklL3C0h9HrsTQrnNyouaZa9KlXZJ72XDCIkIlV0m6EtxbzdMwbH4sLFOpifRlRn+AmzXjxvLovRtn2bXh/X3bUgPxqypaST57Dlpddlk1Mt0oJmGFtwB/FH1JmK21cIC06uXtlUc8lm/9cTQLd5hcEUX+XRrmTdzqxDgRttN8CRfVUAGE7Er+prN4yCIdNtEQdZm8zymEpHTkYplJ/hK7SMf9Iu1k+eCDFYCzvQuzLcJtNpRaGS1BbVA3va5JKrEu96G7a3Wl3DyzmrH8N3+RA+UIHvP6P5v93tI/eTyfMY54rKpLGkfFeeSMAr5aSoUZVYkvFI8xGEcIrqLWPDF91MclLZa7USSVql0wYu1G9KD10IkopeKkTIAl81WfoY5+Kw1o4CHo7bEQ6tfTuTB4IEywf1XKMBYHmsfAe5B9ferkLYtnAzzt1hoiK1m/2CjX8yQAknRLsnAuyeXfJZRZidVKYOKaSDftddbXJpIlJApC"

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[{"type": "text", "text": "What's the weather like in London?"}],
        ),
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "call_386f67af31f9415781bc35071405",
                    "name": "get_weather",
                    "input": {"location": "London"},
                    "provider_specific_fields": {
                        "signature": test_signature,
                    },
                }
            ],
        ),
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    assert len(result) == 2
    assert result[1]["role"] == "assistant"
    assert "tool_calls" in result[1]
    assert len(result[1]["tool_calls"]) == 1

    # Verify thought signature is extracted and placed in provider_specific_fields
    tool_call = result[1]["tool_calls"][0]
    assert tool_call["id"] == "call_386f67af31f9415781bc35071405"
    assert "function" in tool_call
    assert "provider_specific_fields" in tool_call["function"]
    assert (
        tool_call["function"]["provider_specific_fields"]["thought_signature"]
        == test_signature
    )


def test_translate_anthropic_messages_to_openai_tool_result_with_multiple_content_items():
    """
    Test that tool_result with multiple content items creates a single tool message
    (not multiple messages with the same tool_call_id).

    This is a regression test for the bug:
    "each tool_use must have a single result. Found multiple `tool_result` blocks with id"

    When a tool_result has a list of content items (e.g., text + image), we should create
    ONE tool message with combined content, not multiple tool messages with the same ID.
    """

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[{"type": "text", "text": "Take a screenshot and describe it"}],
        ),
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_016hYHBkTf4JDF3p22UoYk5C",
                    "name": "screenshot_tool",
                    "input": {},
                }
            ],
        ),
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_016hYHBkTf4JDF3p22UoYk5C",
                    "content": [
                        {"type": "text", "text": "Here is the screenshot:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                            },
                        },
                        {"type": "text", "text": "Screenshot captured successfully."},
                    ],
                }
            ],
        ),
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    # Count how many tool messages have the same tool_call_id
    tool_messages = [
        msg for msg in result if isinstance(msg, dict) and msg.get("role") == "tool"
    ]
    tool_call_ids = [msg.get("tool_call_id") for msg in tool_messages]

    # The critical assertion: each tool_call_id should appear only ONCE
    assert len(tool_call_ids) == len(set(tool_call_ids)), (
        f"Bug: Found duplicate tool_call_ids! "
        f"Each tool_use must have exactly one tool_result. "
        f"tool_call_ids: {tool_call_ids}"
    )

    # There should be exactly one tool message
    assert len(tool_messages) == 1, f"Expected 1 tool message, got {len(tool_messages)}"

    # The content should be a list with all items combined
    tool_message = tool_messages[0]
    assert tool_message["tool_call_id"] == "toolu_016hYHBkTf4JDF3p22UoYk5C"
    assert isinstance(
        tool_message["content"], list
    ), "Multiple content items should be combined into a list"
    assert (
        len(tool_message["content"]) == 3
    ), f"Expected 3 content items, got {len(tool_message['content'])}"

    # Verify content types
    assert tool_message["content"][0]["type"] == "text"
    assert tool_message["content"][0]["text"] == "Here is the screenshot:"
    assert tool_message["content"][1]["type"] == "image_url"
    assert tool_message["content"][2]["type"] == "text"
    assert tool_message["content"][2]["text"] == "Screenshot captured successfully."


def test_translate_anthropic_messages_to_openai_tool_result_single_item_backward_compat():
    """
    Test that tool_result with a single content item maintains backward compatibility
    by returning a string content (not a list).
    """

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[{"type": "text", "text": "Get the weather"}],
        ),
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_single_item",
                    "name": "get_weather",
                    "input": {"location": "Boston"},
                }
            ],
        ),
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_single_item",
                    "content": [
                        {"type": "text", "text": "72°F and sunny"},
                    ],
                }
            ],
        ),
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    tool_messages = [
        msg for msg in result if isinstance(msg, dict) and msg.get("role") == "tool"
    ]

    assert len(tool_messages) == 1
    tool_message = tool_messages[0]

    # Single item should be a string for backward compatibility
    assert isinstance(tool_message["content"], str), (
        f"Single content item should be a string for backward compatibility, "
        f"got {type(tool_message['content'])}"
    )
    assert tool_message["content"] == "72°F and sunny"


def test_streaming_chunk_with_both_text_and_tool_calls_issue_18238():
    """
    When a streaming choice contains both text content and tool_calls,
    both should be processed (tool_calls should not be ignored).
    """
    # streaming choice with both text and tool_calls
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                provider_specific_fields=None,
                content="Here is some text for litellm",
                role=None,
                function_call=None,
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id="toolu_bdrk_013xRVejhv3ybmLEGCoZib2b",
                        function=Function(arguments='{"cmd": "init"}', name="Bash"),
                        type="function",
                        index=0,
                    )
                ],
                audio=None,
            ),
            logprobs=None,
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()

    # When both text and tool_calls exist, tool_calls (input_json_delta) takes priority
    (
        type_of_content,
        content_block_delta,
    ) = adapter._translate_streaming_openai_chunk_to_anthropic(choices=choices)

    assert type_of_content == "input_json_delta"
    assert content_block_delta["partial_json"] == '{"cmd": "init"}'

    # When both text and tool_calls exist, tool_use should be detected and tool name captured
    (
        block_type,
        content_block_start,
    ) = adapter._translate_streaming_openai_chunk_to_anthropic_content_block(
        choices=choices
    )

    assert block_type == "tool_use"
    assert content_block_start["name"] == "Bash"
    assert content_block_start["id"] == "toolu_bdrk_013xRVejhv3ybmLEGCoZib2b"


# ============================================================================
# Cache Control Transformation Tests
# ============================================================================

# Model constant for cache control tests
CACHE_CONTROL_BEDROCK_CONVERSE_MODEL = "bedrock/converse/global.anthropic.claude-opus-4-5-20251101-v1:0"
CACHE_CONTROL_NON_ANTHROPIC_MODEL = "gpt-4"


def test_should_add_cache_control_for_anthropic_model():
    """Should add cache_control to target for Anthropic Claude models."""
    adapter = LiteLLMAnthropicMessagesAdapter()
    cache_control = {"type": "ephemeral"}

    for model in [
        CACHE_CONTROL_BEDROCK_CONVERSE_MODEL,
        "anthropic/claude-sonnet-4-5",
        "claude-opus-4-5-20251101",
        "vertex_ai/claude-3-sonnet@20240229",
    ]:
        target = {}
        adapter._add_cache_control_if_applicable({"cache_control": cache_control}, target, model)
        assert "cache_control" in target
        assert target["cache_control"] == cache_control


def test_should_not_add_cache_control_for_non_anthropic_model():
    """Should not add cache_control for non-Anthropic models."""
    adapter = LiteLLMAnthropicMessagesAdapter()
    cache_control = {"type": "ephemeral"}

    for model in [CACHE_CONTROL_NON_ANTHROPIC_MODEL, "openai/gpt-4-turbo", "gemini-pro"]:
        target = {}
        adapter._add_cache_control_if_applicable({"cache_control": cache_control}, target, model)
        assert "cache_control" not in target


def test_should_not_add_cache_control_when_none():
    """Should not add cache_control when source has None or empty cache_control."""
    adapter = LiteLLMAnthropicMessagesAdapter()

    for source in [{"cache_control": None}, {"cache_control": {}}, {"cache_control": ""}, {}]:
        target = {}
        adapter._add_cache_control_if_applicable(source, target, CACHE_CONTROL_BEDROCK_CONVERSE_MODEL)
        assert "cache_control" not in target


def test_should_not_add_cache_control_when_model_none():
    """Should not add cache_control when model is None or empty."""
    adapter = LiteLLMAnthropicMessagesAdapter()
    cache_control = {"type": "ephemeral"}

    for model in [None, ""]:
        target = {}
        adapter._add_cache_control_if_applicable({"cache_control": cache_control}, target, model)
        assert "cache_control" not in target


def test_cache_control_preserved_in_text_content_for_claude():
    """Cache control should be preserved in text content for Claude models."""
    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "text",
                    "text": "This is cached content",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(
        messages=anthropic_messages, model=CACHE_CONTROL_BEDROCK_CONVERSE_MODEL
    )

    assert len(result) == 1
    assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_cache_control_not_preserved_for_non_claude_model():
    """Cache control should NOT be preserved for non-Claude models."""
    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "text",
                    "text": "This is cached content",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(
        messages=anthropic_messages, model=CACHE_CONTROL_NON_ANTHROPIC_MODEL
    )

    assert len(result) == 1
    assert "cache_control" not in result[0]["content"][0]


def test_cache_control_preserved_in_image_content_for_claude():
    """Cache control should be preserved in image content for Claude models."""
    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    },
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(
        messages=anthropic_messages, model=CACHE_CONTROL_BEDROCK_CONVERSE_MODEL
    )

    assert len(result) == 1
    assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_cache_control_preserved_in_document_content_for_claude():
    """Cache control should be preserved in document content for Claude models."""
    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "JVBERi0xLjQKJeLjz9MK",
                    },
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(
        messages=anthropic_messages, model=CACHE_CONTROL_BEDROCK_CONVERSE_MODEL
    )

    assert len(result) == 1
    assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_cache_control_preserved_in_tool_result_for_claude():
    """Cache control should be preserved in tool_result for Claude models."""
    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01234",
                    "content": "Tool result content",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(
        messages=anthropic_messages, model=CACHE_CONTROL_BEDROCK_CONVERSE_MODEL
    )

    tool_message = next(msg for msg in result if msg.get("role") == "tool")
    assert tool_message["cache_control"] == {"type": "ephemeral"}


def test_cache_control_not_preserved_in_tool_result_for_non_claude():
    """Cache control should NOT be preserved in tool_result for non-Claude models."""
    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01234",
                    "content": "Tool result content",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(
        messages=anthropic_messages, model=CACHE_CONTROL_NON_ANTHROPIC_MODEL
    )

    tool_message = next(msg for msg in result if msg.get("role") == "tool")
    assert "cache_control" not in tool_message


def test_cache_control_preserved_in_assistant_text_for_claude():
    """Cache control should be preserved in assistant text blocks for Claude models."""
    anthropic_messages = [
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "text",
                    "text": "Assistant response",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(
        messages=anthropic_messages, model=CACHE_CONTROL_BEDROCK_CONVERSE_MODEL
    )

    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    # When cache_control is present, content should be a list
    assert isinstance(result[0]["content"], list)
    assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_cache_control_preserved_in_tool_use_for_claude():
    """Cache control should be preserved in tool_use blocks for Claude models."""
    anthropic_messages = [
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_01234",
                    "name": "get_weather",
                    "input": {"location": "Boston"},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(
        messages=anthropic_messages, model=CACHE_CONTROL_BEDROCK_CONVERSE_MODEL
    )

    assert len(result) == 1
    assert "tool_calls" in result[0]
    assert result[0]["tool_calls"][0]["cache_control"] == {"type": "ephemeral"}


def test_cache_control_preserved_in_tools_for_claude():
    """Cache control should be preserved in tools for Claude models."""
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather for a location",
            "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}},
            "cache_control": {"type": "ephemeral"},
        }
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result, tool_name_mapping = adapter.translate_anthropic_tools_to_openai(
        tools=tools, model=CACHE_CONTROL_BEDROCK_CONVERSE_MODEL
    )

    assert len(result) == 1
    assert result[0]["cache_control"] == {"type": "ephemeral"}
    assert tool_name_mapping == {}  # No truncation needed for short names


def test_cache_control_not_preserved_in_tools_for_non_claude():
    """Cache control should NOT be preserved in tools for non-Claude models."""
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather for a location",
            "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}},
            "cache_control": {"type": "ephemeral"},
        }
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result, tool_name_mapping = adapter.translate_anthropic_tools_to_openai(
        tools=tools, model=CACHE_CONTROL_NON_ANTHROPIC_MODEL
    )

    assert len(result) == 1
    assert "cache_control" not in result[0]


def test_translate_openai_content_to_anthropic_reasoning_content_without_thinking_blocks():
    """
    Test that reasoning_content is converted to thinking block when thinking_blocks is not present.
    This handles providers like OpenRouter that return reasoning_content instead of thinking_blocks.
    
    Regression test for: OpenRouter models returning reasoning_content in /v1/messages endpoint
    should be converted to Anthropic's thinking block format.
    """
    openai_choices = [
        Choices(
            message=Message(
                role="assistant",
                content="There are **3** \"r\"s in the word strawberry.",
                reasoning_content="**Considering Letter Frequency**\n\nI've homed in on the specifics: The task focuses on counting the letter 'r'. I've identified the target word, \"strawberry,\" and confirmed my understanding of the letter's location. The first 'r' follows 't', the second after 'e', and the third… well, I'm almost there.\n\n\n**Calculating the Count**\n\nMy analysis is complete! I've confirmed that the letter \"r\" appears three times in \"strawberry.\" The first follows \"t,\" the second \"e,\" and the third immediately follows the second. The count is definitively three.",
            )
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter._translate_openai_content_to_anthropic(choices=openai_choices)

    assert len(result) == 2
    # First block should be thinking block with reasoning_content
    assert result[0]["type"] == "thinking"
    assert "Considering Letter Frequency" in result[0]["thinking"]
    assert "Calculating the Count" in result[0]["thinking"]
    assert result[0]["signature"] is None
    # Second block should be text block with content
    assert result[1]["type"] == "text"
    assert result[1]["text"] == "There are **3** \"r\"s in the word strawberry."


def test_translate_streaming_openai_chunk_to_anthropic_reasoning_content_without_thinking_blocks():
    """
    Test that reasoning_content in streaming chunks is converted to thinking_delta 
    when thinking_blocks is not present.
    
    This handles providers like OpenRouter that return reasoning_content in streaming 
    responses without thinking_blocks.
    """
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                reasoning_content="I need to analyze this carefully...",
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
    assert content_block_delta["thinking"] == "I need to analyze this carefully..."


def test_translate_openai_response_to_anthropic_with_reasoning_content_only():
    """
    Test the full response translation when only reasoning_content is present 
    (no thinking_blocks).
    
    This simulates OpenRouter's response format being translated to Anthropic format
    through /v1/messages endpoint.
    """
    openai_response = ModelResponse(
        id="gen-1770027855-HyrqYvLcX8oTLNgfyDob",
        model="gemini-3-flash",
        choices=[
            Choices(
                finish_reason="stop",
                message=Message(
                    role="assistant",
                    content="There are **3** \"r\"s in the word strawberry.",
                    reasoning_content="**Considering Letter Frequency**\n\nI've homed in on the specifics: The task focuses on counting the letter 'r'.",
                ),
            )
        ],
        usage=Usage(prompt_tokens=13, completion_tokens=138),
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    anthropic_response = adapter.translate_openai_response_to_anthropic(
        response=openai_response
    )

    anthropic_content = anthropic_response.get("content")
    assert anthropic_content is not None
    assert len(anthropic_content) == 2
    
    # First block should be thinking
    assert anthropic_content[0]["type"] == "thinking"
    assert "Considering Letter Frequency" in anthropic_content[0]["thinking"]
    assert anthropic_content[0].get("signature") is None
    
    # Second block should be text
    assert anthropic_content[1]["type"] == "text"
    assert anthropic_content[1]["text"] == "There are **3** \"r\"s in the word strawberry."
    
    assert anthropic_response.get("stop_reason") == "end_turn"


# =====================================================================
# Tool Name Truncation Tests (Issue #17904)
# OpenAI has a 64-character limit for function/tool names
# =====================================================================


def test_truncate_tool_name_short_name():
    """Short tool names should not be truncated."""
    short_name = "get_weather"
    result = truncate_tool_name(short_name)
    assert result == short_name
    assert len(result) <= OPENAI_MAX_TOOL_NAME_LENGTH


def test_truncate_tool_name_exactly_64_chars():
    """Tool names exactly 64 chars should not be truncated."""
    name_64_chars = "a" * 64
    result = truncate_tool_name(name_64_chars)
    assert result == name_64_chars
    assert len(result) == 64


def test_truncate_tool_name_long_name():
    """Long tool names should be truncated with hash suffix."""
    long_name = "computer_tool_with_very_long_name_that_exceeds_openai_64_character_limit_and_keeps_going"
    result = truncate_tool_name(long_name)

    assert len(result) == OPENAI_MAX_TOOL_NAME_LENGTH
    assert result != long_name
    # Should have format: {55-char-prefix}_{8-char-hash}
    assert "_" in result
    parts = result.rsplit("_", 1)
    assert len(parts[0]) == 55
    assert len(parts[1]) == 8


def test_truncate_tool_name_deterministic():
    """Truncation should be deterministic (same input = same output)."""
    long_name = "a_very_long_tool_name_that_needs_to_be_truncated_for_openai_compatibility_reasons"
    result1 = truncate_tool_name(long_name)
    result2 = truncate_tool_name(long_name)
    assert result1 == result2


def test_truncate_tool_name_avoids_collisions():
    """Similar long names should produce different truncated names."""
    name1 = "process_user_data_with_validation_and_error_handling_for_production_environment"
    name2 = "process_user_data_with_validation_and_error_handling_for_staging_environment"

    result1 = truncate_tool_name(name1)
    result2 = truncate_tool_name(name2)

    assert result1 != result2  # Different hashes prevent collision


def test_create_tool_name_mapping_no_long_names():
    """Mapping should be empty when no names need truncation."""
    tools = [
        {"name": "get_weather"},
        {"name": "search_web"},
    ]
    mapping = create_tool_name_mapping(tools)
    assert mapping == {}


def test_create_tool_name_mapping_with_long_names():
    """Mapping should contain entries for truncated names."""
    long_name = "a_very_long_tool_name_that_exceeds_the_64_character_limit_imposed_by_openai"
    tools = [
        {"name": "short_name"},
        {"name": long_name},
    ]
    mapping = create_tool_name_mapping(tools)

    assert len(mapping) == 1
    truncated = truncate_tool_name(long_name)
    assert truncated in mapping
    assert mapping[truncated] == long_name


def test_translate_anthropic_tools_with_long_names():
    """Tools with long names should be truncated and mapped."""
    long_name = "computer_tool_with_very_long_descriptive_name_that_exceeds_openai_limit_completely"
    tools = [
        {
            "name": long_name,
            "description": "A tool with a very long name",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result, tool_name_mapping = adapter.translate_anthropic_tools_to_openai(
        tools=tools, model="gpt-4"
    )

    assert len(result) == 1
    # The tool name should be truncated
    truncated_name = result[0]["function"]["name"]
    assert len(truncated_name) <= 64
    assert truncated_name != long_name
    # Mapping should have the reverse lookup
    assert truncated_name in tool_name_mapping
    assert tool_name_mapping[truncated_name] == long_name


def test_translate_anthropic_tools_mixed_names():
    """Mix of short and long names should work correctly."""
    short_name = "get_weather"
    long_name = "process_complex_data_transformation_with_validation_and_error_handling_pipeline"
    tools = [
        {"name": short_name, "input_schema": {"type": "object"}},
        {"name": long_name, "input_schema": {"type": "object"}},
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result, tool_name_mapping = adapter.translate_anthropic_tools_to_openai(
        tools=tools, model="gpt-4"
    )

    assert len(result) == 2
    # Short name unchanged
    assert result[0]["function"]["name"] == short_name
    # Long name truncated
    assert result[1]["function"]["name"] != long_name
    assert len(result[1]["function"]["name"]) <= 64
    # Only long name in mapping
    assert len(tool_name_mapping) == 1


def test_translate_openai_response_restores_tool_names():
    """Tool names in responses should be restored to original."""
    original_name = "a_very_long_tool_name_that_needs_truncation_for_openai_api_compatibility"
    truncated_name = truncate_tool_name(original_name)
    tool_name_mapping = {truncated_name: original_name}

    # Create a mock OpenAI response with the truncated name
    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                index=0,
                finish_reason="tool_calls",
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionAssistantToolCall(
                            id="call_123",
                            type="function",
                            function=Function(
                                name=truncated_name,
                                arguments='{"arg": "value"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        model="gpt-4",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_openai_response_to_anthropic(
        response=response, tool_name_mapping=tool_name_mapping
    )

    # Find the tool_use block in the response
    tool_use_blocks = [c for c in result["content"] if c.get("type") == "tool_use"]
    assert len(tool_use_blocks) == 1
    # Name should be restored to original
    assert tool_use_blocks[0]["name"] == original_name


def test_translate_openai_response_to_anthropic_input_tokens_excludes_cached_tokens():
    """
    Regression test: input_tokens in Anthropic format should NOT include cached tokens.
    
    Issue: v1/messages API was returning incorrect input_token count when using prompt caching.
    The OpenAI format includes cached tokens in prompt_tokens, but Anthropic format should not.
    
    According to Anthropic's spec:
    - input_tokens = uncached input tokens only
    - cache_read_input_tokens = tokens read from cache
    
    In OpenAI format:
    - prompt_tokens = all input tokens (including cached)
    - prompt_tokens_details.cached_tokens = cached tokens
    
    Expected: anthropic.input_tokens = openai.prompt_tokens - openai.prompt_tokens_details.cached_tokens
    """
    from litellm.types.utils import PromptTokensDetailsWrapper

    # Create OpenAI format response with cached tokens
    # Scenario: 100 total prompt tokens, 30 of which are cached
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=30
        ),
        cache_read_input_tokens=30,  # Anthropic format cache info
    )
    
    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=Message(
                    role="assistant",
                    content="Test response",
                ),
            )
        ],
        model="claude-3-sonnet-20240229",
        usage=usage,
    )
    
    # Convert to Anthropic format
    adapter = LiteLLMAnthropicMessagesAdapter()
    anthropic_response = adapter.translate_openai_response_to_anthropic(
        response=response,
        tool_name_mapping=None,
    )
    
    # Validate: input_tokens should be 70 (100 - 30 cached), not 100
    assert anthropic_response["usage"]["input_tokens"] == 70, (
        f"Expected input_tokens=70 (100 total - 30 cached), "
        f"but got {anthropic_response['usage']['input_tokens']}. "
        f"input_tokens should NOT include cached tokens per Anthropic spec."
    )
    assert anthropic_response["usage"]["output_tokens"] == 50
    assert anthropic_response["usage"]["cache_read_input_tokens"] == 30


def test_translate_openai_response_to_anthropic_input_tokens_no_cache():
    """
    Regression test: input_tokens should equal prompt_tokens when there are no cached tokens.
    """
    from litellm.types.utils import PromptTokensDetailsWrapper

    # Create OpenAI format response without cached tokens
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
    
    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=Message(
                    role="assistant",
                    content="Test response",
                ),
            )
        ],
        model="claude-3-sonnet-20240229",
        usage=usage,
    )
    
    # Convert to Anthropic format
    adapter = LiteLLMAnthropicMessagesAdapter()
    anthropic_response = adapter.translate_openai_response_to_anthropic(
        response=response,
        tool_name_mapping=None,
    )
    
    # Validate: input_tokens should equal prompt_tokens when no caching
    assert anthropic_response["usage"]["input_tokens"] == 100
    assert anthropic_response["usage"]["output_tokens"] == 50
