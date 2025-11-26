import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))


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
    assert tool_call["function"]["provider_specific_fields"]["thought_signature"] == test_signature
