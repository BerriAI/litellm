import json
from unittest.mock import patch

import pytest

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import (
    BAD_MESSAGE_ERROR_STR,
    BedrockConverseMessagesProcessor,
    BedrockImageProcessor,
    ollama_pt,
)


def test_ollama_pt_simple_messages():
    """Test basic functionality with simple text messages"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "assistant", "content": "How can I help you?"},
        {"role": "user", "content": "Hello"},
    ]

    result = ollama_pt(model="llama2", messages=messages)

    expected_prompt = "### System:\nYou are a helpful assistant\n\n### Assistant:\nHow can I help you?\n\n### User:\nHello\n\n"
    assert isinstance(result, dict)
    assert result["prompt"] == expected_prompt
    assert result["images"] == []


def test_ollama_pt_consecutive_user_messages():
    """Test handling consecutive user messages"""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "How can I help you?"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm good, thanks!"},
        {"role": "user", "content": "I am well too."},
    ]

    result = ollama_pt(model="llama2", messages=messages)

    # Consecutive user messages should be merged
    expected_prompt = "### User:\nHello\n\n### Assistant:\nHow can I help you?\n\n### User:\nHow are you?\n\n### Assistant:\nI'm good, thanks!\n\n### User:\nI am well too.\n\n"
    assert isinstance(result, dict)
    assert result["prompt"] == expected_prompt


@pytest.mark.asyncio
async def test_anthropic_bedrock_thinking_blocks_with_none_content():
    """
    Test the specific function that processes thinking_blocks when content is None
    """
    mock_assistant_message = {
        "content": "None",  # content is None
        "role": "assistant",
        "thinking_blocks": [
            {
                "type": "thinking",
                "thinking": "This is a test thinking block",
                "signature": "test-signature",
            }
        ],
        "reasoning_content": "This is the reasoning content",
    }
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        mock_assistant_message,
    ]

    # test _bedrock_converse_messages_pt_async
    result = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages=messages,
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        llm_provider="bedrock",
    )

    # verify the result
    assert len(result) == 2
    assert (
        result[1]["content"][0]["reasoningContent"]["reasoningText"]["text"]
        == "This is a test thinking block"
    )


def test_bedrock_validate_format_image_or_video():
    """Test the _validate_format method for images, videos, and documents"""

    # Test valid image formats
    valid_image_formats = ["png", "jpeg", "gif", "webp"]
    for format in valid_image_formats:
        result = BedrockImageProcessor._validate_format(f"image/{format}", format)
        assert result == format, f"Expected {format}, got {result}"

    # Test valid video formats
    valid_video_formats = [
        "mp4",
        "mov",
        "mkv",
        "webm",
        "flv",
        "mpeg",
        "mpg",
        "wmv",
        "3gp",
    ]
    for format in valid_video_formats:
        result = BedrockImageProcessor._validate_format(f"video/{format}", format)
        assert result == format, f"Expected {format}, got {result}"


# def test_ollama_pt_consecutive_system_messages():
#     """Test handling consecutive system messages"""
#     messages = [
#         {"role": "user", "content": "Hello"},
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "system", "content": "Be concise and polite"},
#         {"role": "assistant", "content": "How can I help you?"}
#     ]

#     result = ollama_pt(model="llama2", messages=messages)

#     # Consecutive system messages should be merged
#     expected_prompt = "### User:\nHello\n\n### System:\nYou are a helpful assistantBe concise and polite\n\n### Assistant:\nHow can I help you?\n\n"
#     assert result == expected_prompt

# def test_ollama_pt_consecutive_assistant_messages():
#     """Test handling consecutive assistant messages"""
#     messages = [
#         {"role": "user", "content": "Hello"},
#         {"role": "assistant", "content": "Hi there!"},
#         {"role": "assistant", "content": "How can I help you?"},
#         {"role": "user", "content": "Tell me a joke"}
#     ]

#     result = ollama_pt(model="llama2", messages=messages)

#     # Consecutive assistant messages should be merged
#     expected_prompt = "### User:\nHello\n\n### Assistant:\nHi there!How can I help you?\n\n### User:\nTell me a joke\n\n"
#     assert result["prompt"] == expected_prompt

# def test_ollama_pt_with_image_urls_as_strings():
#     """Test handling messages with image URLs as strings"""
#     messages = [
#         {"role": "user", "content": [
#             {"type": "text", "text": "What's in this image?"},
#             {"type": "image_url", "image_url": "http://example.com/image.jpg"}
#         ]},
#         {"role": "assistant", "content": "That's a cat."}
#     ]

#     result = ollama_pt(model="llama2", messages=messages)

#     expected_prompt = "### User:\nWhat's in this image?\n\n### Assistant:\nThat's a cat.\n\n"
#     assert result["prompt"] == expected_prompt
#     assert result["images"] == ["http://example.com/image.jpg"]

# def test_ollama_pt_with_image_urls_as_dicts():
#     """Test handling messages with image URLs as dictionaries"""
#     messages = [
#         {"role": "user", "content": [
#             {"type": "text", "text": "What's in this image?"},
#             {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
#         ]},
#         {"role": "assistant", "content": "That's a cat."}
#     ]

#     result = ollama_pt(model="llama2", messages=messages)

#     expected_prompt = "### User:\nWhat's in this image?\n\n### Assistant:\nThat's a cat.\n\n"
#     assert result["prompt"] == expected_prompt
#     assert result["images"] == ["http://example.com/image.jpg"]

# def test_ollama_pt_with_tool_calls():
#     """Test handling messages with tool calls"""
#     messages = [
#         {"role": "user", "content": "What's the weather in San Francisco?"},
#         {"role": "assistant", "content": "I'll check the weather for you.",
#          "tool_calls": [
#              {
#                  "id": "call_123",
#                  "type": "function",
#                  "function": {
#                      "name": "get_weather",
#                      "arguments": json.dumps({"location": "San Francisco"})
#                  }
#              }
#          ]
#         },
#         {"role": "tool", "content": "Sunny, 72°F"}
#     ]

#     result = ollama_pt(model="llama2", messages=messages)

#     # Check if tool call is included in the prompt
#     assert "### User:\nWhat's the weather in San Francisco?" in result["prompt"]
#     assert "### Assistant:\nI'll check the weather for you.Tool Calls:" in result["prompt"]
#     assert "get_weather" in result["prompt"]
#     assert "San Francisco" in result["prompt"]
#     assert "### User:\nSunny, 72°F\n\n" in result["prompt"]

# def test_ollama_pt_error_handling():
#     """Test error handling for invalid messages"""
#     messages = [
#         {"role": "invalid_role", "content": "This is an invalid role"}
#     ]

#     with pytest.raises(litellm.BadRequestError) as excinfo:
#         ollama_pt(model="llama2", messages=messages)

#     assert BAD_MESSAGE_ERROR_STR in str(excinfo.value)

# def test_ollama_pt_empty_messages():
#     """Test with empty messages list"""
#     messages = []

#     result = ollama_pt(model="llama2", messages=messages)

#     assert result["prompt"] == ""
#     assert result["images"] == []

# def test_ollama_pt_with_tool_message_content():
#     """Test handling tool message content"""
#     messages = [
#         {"role": "user", "content": "Tell me a joke"},
#         {"role": "assistant", "content": "Why did the chicken cross the road?"},
#         {"role": "user", "content": "Why?"},
#         {"role": "assistant", "content": "To get to the other side!"},
#         {"role": "tool", "content": "Joke rating: 5/10"}
#     ]

#     result = ollama_pt(model="llama2", messages=messages)

#     assert "### User:\nTell me a joke" in result["prompt"]
#     assert "### Assistant:\nWhy did the chicken cross the road?" in result["prompt"]
#     assert "### User:\nWhy?" in result["prompt"]
#     assert "### Assistant:\nTo get to the other side!" in result["prompt"]
#     assert "### User:\nJoke rating: 5/10\n\n" in result["prompt"]

# def test_ollama_pt_with_function_message():
#     """Test handling function messages (treated as user message type)"""
#     messages = [
#         {"role": "user", "content": "What's 2+2?"},
#         {"role": "function", "content": "The result is 4"},
#         {"role": "assistant", "content": "The answer is 4."}
#     ]

#     result = ollama_pt(model="llama2", messages=messages)

#     assert "### User:\nWhat's 2+2?The result is 4\n\n" in result["prompt"]
#     assert "### Assistant:\nThe answer is 4.\n\n" in result["prompt"]

# def test_ollama_pt_with_multiple_images():
#     """Test handling multiple images in a message"""
#     messages = [
#         {"role": "user", "content": [
#             {"type": "text", "text": "Compare these images:"},
#             {"type": "image_url", "image_url": "http://example.com/image1.jpg"},
#             {"type": "image_url", "image_url": "http://example.com/image2.jpg"}
#         ]},
#         {"role": "assistant", "content": "Both images show cats, but different breeds."}
#     ]

#     result = ollama_pt(model="llama2", messages=messages)

#     expected_prompt = "### User:\nCompare these images:\n\n### Assistant:\nBoth images show cats, but different breeds.\n\n"
#     assert result["prompt"] == expected_prompt
#     assert result["images"] == ["http://example.com/image1.jpg", "http://example.com/image2.jpg"]

# def test_ollama_pt_mixed_content_types():
#     """Test handling a mix of string and list content types"""
#     messages = [
#         {"role": "user", "content": "Hello"},
#         {"role": "assistant", "content": "Hi there!"},
#         {"role": "user", "content": [
#             {"type": "text", "text": "Look at this:"},
#             {"type": "image_url", "image_url": "http://example.com/image.jpg"}
#         ]},
#         {"role": "system", "content": "Be helpful"},
#         {"role": "assistant", "content": "I see a cat in the image."}
#     ]

#     result = ollama_pt(model="llama2", messages=messages)

#     assert "### User:\nHello\n\n" in result["prompt"]
#     assert "### Assistant:\nHi there!\n\n" in result["prompt"]
#     assert "### User:\nLook at this:\n\n" in result["prompt"]
#     assert "### System:\nBe helpful\n\n" in result["prompt"]
#     assert "### Assistant:\nI see a cat in the image.\n\n" in result["prompt"]
#     assert result["images"] == ["http://example.com/image.jpg"]
