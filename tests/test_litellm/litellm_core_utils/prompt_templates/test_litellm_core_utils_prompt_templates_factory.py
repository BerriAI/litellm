import base64
import json
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import (
    BAD_MESSAGE_ERROR_STR,
    BedrockConverseMessagesProcessor,
    BedrockImageProcessor,
    _bedrock_converse_messages_pt,
    _convert_to_bedrock_tool_call_invoke,
    _convert_to_bedrock_tool_call_result,
    anthropic_messages_pt,
    convert_to_gemini_tool_call_result,
    ollama_pt,
    sanitize_messages_for_tool_calling,
)
from litellm.types.llms.openai import ChatCompletionToolMessage


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
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        llm_provider="bedrock",
    )

    # verify the result
    assert len(result) == 2
    assert (
        result[1]["content"][0]["reasoningContent"]["reasoningText"]["text"]
        == "This is a test thinking block"
    )


def test_bedrock_converse_assistant_with_empty_thinking_block_and_tool_calls():
    """
    Regression: Claude Code (with extended thinking enabled) replays prior
    assistant turns that include an empty thinking block alongside tool_use
    blocks, e.g.

        content=[
            {"type": "text", "text": ""},
            {"type": "thinking", "thinking": "", "signature": ""},
            {"type": "tool_use", ...},
        ]

    After the Anthropic→OpenAI adapter, this becomes assistant message with
    content="" and thinking_blocks=[{thinking:"", signature:""}] plus
    tool_calls. The Bedrock Converse fallback for unsigned reasoning content
    was emitting `BedrockContentBlock(text="")`, which Bedrock rejects with:

        "The text field in the ContentBlock object at messages.X.content.0
         is blank."

    Verify no blank-text ContentBlocks are produced.
    """
    messages = [
        {"role": "user", "content": "tell me about this repo"},
        {
            "role": "assistant",
            "content": "",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "", "signature": ""},
            ],
            "tool_calls": [
                {
                    "id": "tooluse_aC8Izm8kl5DqVkgLA4XqcH",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": '{"command": "ls"}'},
                },
                {
                    "id": "tooluse_31BEsgAjDwZxsUofwmdVPS",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": '{"command": "pwd"}'},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tooluse_aC8Izm8kl5DqVkgLA4XqcH",
            "content": "file1\nfile2",
        },
        {
            "role": "tool",
            "tool_call_id": "tooluse_31BEsgAjDwZxsUofwmdVPS",
            "content": "/repo",
        },
    ]

    result = _bedrock_converse_messages_pt(
        messages=messages,
        model="us.anthropic.claude-opus-4-7",
        llm_provider="bedrock",
    )

    assistant_blocks = [m for m in result if m["role"] == "assistant"]
    assert len(assistant_blocks) == 1
    for block in assistant_blocks[0]["content"]:
        if "text" in block:
            assert block[
                "text"
            ].strip(), (
                f"Bedrock Converse rejects blank-text ContentBlocks; got {block!r}"
            )
    # toolUse blocks must still be present
    tool_use_blocks = [b for b in assistant_blocks[0]["content"] if "toolUse" in b]
    assert len(tool_use_blocks) == 2


def test_convert_to_azure_openai_messages():
    """Test coverting image_url to azure_openai spec"""

    from typing import List

    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_azure_openai_messages,
    )
    from litellm.types.llms.openai import AllMessageValues

    input: List[AllMessageValues] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": "www.mock.com"},
            ],
        }
    ]

    expected_content = [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "www.mock.com"}},
    ]

    output = convert_to_azure_openai_messages(input)

    content = output[0].get("content")
    assert content == expected_content


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

    # Test valid document formats
    valid_document_formats = {
        "application/pdf": "pdf",
        "text/csv": "csv",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    }
    for mime, expected in valid_document_formats.items():
        print("testing mime", mime, "expected", expected)
        result = BedrockImageProcessor._validate_format(mime, mime.split("/")[1])
        assert result == expected, f"Expected {expected}, got {result}"


def test_bedrock_get_document_format_fallback_mimes():
    """
    Test the _get_document_format method with fallback MIME types for DOCX and XLSX.

    This tests the fallback mechanism when mimetypes.guess_all_extensions returns empty results,
    which can happen in Docker containers where mimetypes depends on OS-installed MIME types.
    """
    from unittest.mock import patch

    # Test DOCX fallback
    docx_mime = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    supported_formats = ["pdf", "docx", "xlsx", "csv"]

    # Mock mimetypes.guess_all_extensions to return empty list (simulating Docker container scenario)
    with patch("mimetypes.guess_all_extensions", return_value=[]):
        result = BedrockImageProcessor._get_document_format(
            mime_type=docx_mime, supported_doc_formats=supported_formats
        )
        assert result == "docx", f"Expected 'docx', got '{result}'"

    # Test XLSX fallback
    xlsx_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    with patch("mimetypes.guess_all_extensions", return_value=[]):
        result = BedrockImageProcessor._get_document_format(
            mime_type=xlsx_mime, supported_doc_formats=supported_formats
        )
        assert result == "xlsx", f"Expected 'xlsx', got '{result}'"


def test_bedrock_get_document_format_mimetypes_success():
    """
    Test the _get_document_format method when mimetypes.guess_all_extensions works normally.
    """
    docx_mime = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    supported_formats = ["pdf", "docx", "xlsx", "csv"]

    # Test normal mimetypes behavior (should not hit fallback)
    result = BedrockImageProcessor._get_document_format(
        mime_type=docx_mime, supported_doc_formats=supported_formats
    )
    assert result == "docx", f"Expected 'docx', got '{result}'"


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


def test_vertex_ai_transform_empty_function_call_arguments():
    """
    Test that the _transform_parts method handles empty function call arguments correctly
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        VertexFunctionCall,
        _gemini_tool_call_invoke_helper,
    )

    function_call = {
        "name": "get_weather",
        "arguments": "",
    }
    result: VertexFunctionCall = _gemini_tool_call_invoke_helper(function_call)
    print(result)
    assert result["args"] == {
        "type": "object",
    }


@pytest.mark.asyncio
async def test_bedrock_process_image_async_factory():
    """
    Test that the _process_image_async_factory method handles image input correctly
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        BedrockImageProcessor,
    )

    image_url = "data:application/pdf; qs=0.001;base64,JVBERi0xLjQKJcOkw7zDtsOfCjIgMCBvYmoKPDwvTGVuZ3RoIDMgMCBSL0ZpbHRlci9GbGF0ZURlY29kZT4"

    content_block = await BedrockImageProcessor.process_image_async(
        image_url=image_url, format=None
    )
    print(f"content_block: {content_block}")


def test_unpack_defs_resolves_nested_ref_inside_anyof_items():
    """Ensure unpack_defs correctly resolves $ref inside items within anyOf (Issue #11372)."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import unpack_defs

    # Define a minimal schema reproducing the bug scenario
    schema = {
        "type": "object",
        "properties": {
            "vatAmounts": {
                "anyOf": [
                    {  # List of VatAmount
                        "type": "array",
                        "items": {"$ref": "#/$defs/VatAmount"},
                    },
                    {"type": "null"},
                ],
                "title": "Vat Amounts",
            }
        },
        "$defs": {
            "VatAmount": {
                "type": "object",
                "properties": {
                    "vatRate": {"type": "number"},
                    "vatAmount": {"type": "number"},
                },
                "required": ["vatRate", "vatAmount"],
                "title": "VatAmount",
            }
        },
    }

    # Perform unpacking
    unpack_defs(schema, schema["$defs"])

    # Extract the items schema after unpacking
    items_schema = schema["properties"]["vatAmounts"]["anyOf"][0]["items"]

    # Assertions: items_schema should now be the resolved object, not an empty dict
    assert isinstance(
        items_schema, dict
    ), "Items schema should be a dict after unpacking"
    assert items_schema.get("type") == "object"
    # Ensure essential properties are present
    assert set(items_schema.get("properties", {}).keys()) == {"vatRate", "vatAmount"}


def test_convert_gemini_messages():
    """
    Handle 'content' not being present in the message - https://github.com/BerriAI/litellm/issues/13169
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_gemini_tool_call_result,
    )
    from litellm.types.llms.openai import ChatCompletionToolMessage

    message = ChatCompletionToolMessage(
        role="tool",
        tool_call_id="call_d5b2e3fe-d2c0-451d-b034-cf4fbb22e66c",
    )
    last_message_with_tool_calls = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_d5b2e3fe-d2c0-451d-b034-cf4fbb22e66c",
                "type": "function",
                "index": 0,
                "function": {"name": "tool_MAX_Data__get_issues", "arguments": "{}"},
            }
        ],
    }

    convert_to_gemini_tool_call_result(
        message=message,
        last_message_with_tool_calls=last_message_with_tool_calls,
    )


def test_convert_gemini_tool_call_result_with_image_url():
    """
    Test that image_url content type in tool results is handled correctly for Gemini.
    Fixes: https://github.com/BerriAI/litellm/issues/18187
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_gemini_tool_call_result,
    )
    from litellm.types.llms.openai import ChatCompletionToolMessage

    # Test with string image_url format
    message_str_format = ChatCompletionToolMessage(
        role="tool",
        tool_call_id="call_123",
        content=[{"type": "image_url", "image_url": "data:image/jpeg;base64,/9j/4AAQ"}],
    )
    last_message_with_tool_calls = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "index": 0,
                "function": {"name": "get_image", "arguments": "{}"},
            }
        ],
    }

    result = convert_to_gemini_tool_call_result(
        message=message_str_format,
        last_message_with_tool_calls=last_message_with_tool_calls,
    )
    # Should have inline_data for the image
    assert isinstance(result, list) and any("inline_data" in p for p in result)

    # Test with dict image_url format (OpenAI standard)
    message_dict_format = ChatCompletionToolMessage(
        role="tool",
        tool_call_id="call_456",
        content=[
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"},
            }
        ],
    )
    last_message_with_tool_calls["tool_calls"][0]["id"] = "call_456"

    result2 = convert_to_gemini_tool_call_result(
        message=message_dict_format,
        last_message_with_tool_calls=last_message_with_tool_calls,
    )
    assert isinstance(result2, list) and any("inline_data" in p for p in result2)


def test_convert_gemini_tool_call_result_with_anthropic_image_block():
    """
    Test that Anthropic-native image blocks in tool_result list content are
    converted to Gemini inline_data instead of being silently dropped.
    Fixes: https://github.com/BerriAI/litellm/issues/23712
    """
    tiny_png_b64 = base64.b64encode(b"PNG_PLACEHOLDER").decode()

    message = ChatCompletionToolMessage(
        role="tool",
        tool_call_id="call_123",
        content=[
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": tiny_png_b64,
                },
            }
        ],
    )
    last_message_with_tool_calls = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "index": 0,
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ],
    }

    result = convert_to_gemini_tool_call_result(
        message=message,
        last_message_with_tool_calls=last_message_with_tool_calls,
    )
    assert isinstance(result, list), "expected a list of parts"
    inline_parts = [p for p in result if "inline_data" in p]
    assert len(inline_parts) == 1, "expected exactly one inline_data part"
    assert inline_parts[0]["inline_data"]["mime_type"] == "image/png"
    assert inline_parts[0]["inline_data"]["data"] == tiny_png_b64


def test_convert_gemini_tool_call_result_with_multiple_anthropic_image_blocks():
    """
    Test that multiple Anthropic-native image blocks in a single tool_result
    are all preserved as separate inline_data parts instead of only the last
    one being kept.
    Fixes: https://github.com/BerriAI/litellm/issues/23712
    """
    png_b64 = base64.b64encode(b"PNG_PLACEHOLDER").decode()
    jpeg_b64 = base64.b64encode(b"JPEG_PLACEHOLDER").decode()

    message = ChatCompletionToolMessage(
        role="tool",
        tool_call_id="call_multi",
        content=[
            {"type": "text", "text": "here are two images"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": png_b64,
                },
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": jpeg_b64,
                },
            },
        ],
    )
    last_message_with_tool_calls = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_multi",
                "type": "function",
                "index": 0,
                "function": {"name": "screenshot", "arguments": "{}"},
            }
        ],
    }

    result = convert_to_gemini_tool_call_result(
        message=message,
        last_message_with_tool_calls=last_message_with_tool_calls,
    )
    assert isinstance(result, list), "expected a list of parts"
    inline_parts = [p for p in result if "inline_data" in p]
    assert (
        len(inline_parts) == 2
    ), f"expected 2 inline_data parts, got {len(inline_parts)}"
    mime_types = {p["inline_data"]["mime_type"] for p in inline_parts}
    assert mime_types == {"image/png", "image/jpeg"}


def test_convert_gemini_tool_call_result_with_data_url_string():
    """
    Test that a data-URL string in tool_result content is converted to
    Gemini inline_data instead of being passed as plain text.
    Fixes: https://github.com/BerriAI/litellm/issues/23712
    """
    tiny_png_b64 = base64.b64encode(b"PNG_PLACEHOLDER").decode()

    message = ChatCompletionToolMessage(
        role="tool",
        tool_call_id="call_456",
        content=f"data:image/png;base64,{tiny_png_b64}",
    )
    last_message_with_tool_calls = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_456",
                "type": "function",
                "index": 0,
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ],
    }

    result = convert_to_gemini_tool_call_result(
        message=message,
        last_message_with_tool_calls=last_message_with_tool_calls,
    )
    assert isinstance(result, list), "expected a list of parts"
    inline_parts = [p for p in result if "inline_data" in p]
    assert (
        len(inline_parts) == 1
    ), "data-URL image string was not converted to inline_data"
    assert inline_parts[0]["inline_data"]["mime_type"] == "image/png"
    assert inline_parts[0]["inline_data"]["data"] == tiny_png_b64


def test_convert_gemini_tool_call_result_with_data_url_extra_params():
    """
    Test that a data-URL with extra MIME parameters (e.g. charset) produces
    a clean mime_type without the extra parameters.
    """
    tiny_png_b64 = base64.b64encode(b"PNG_PLACEHOLDER").decode()

    message = ChatCompletionToolMessage(
        role="tool",
        tool_call_id="call_extra",
        content=f"data:image/png;charset=UTF-8;base64,{tiny_png_b64}",
    )
    last_message_with_tool_calls = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_extra",
                "type": "function",
                "index": 0,
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ],
    }

    result = convert_to_gemini_tool_call_result(
        message=message,
        last_message_with_tool_calls=last_message_with_tool_calls,
    )
    assert isinstance(result, list), "expected a list of parts"
    inline_parts = [p for p in result if "inline_data" in p]
    assert len(inline_parts) == 1
    assert (
        inline_parts[0]["inline_data"]["mime_type"] == "image/png"
    ), f"expected clean 'image/png', got '{inline_parts[0]['inline_data']['mime_type']}'"


def test_bedrock_tools_unpack_defs():
    """
    Test that the unpack_defs method handles nested $ref inside anyOf items correctly
    """
    from litellm.litellm_core_utils.prompt_templates.factory import _bedrock_tools_pt

    circularRefSchema = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["doc"]},
            "content": {"type": "array", "items": {"$ref": "#/$defs/node"}},
        },
        "required": ["type", "content"],
        "additionalProperties": False,
        "$defs": {
            "node": {
                "type": "object",
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["bulletList"]},
                            "content": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/listItem"},
                            },
                        },
                        "required": ["type"],
                        "additionalProperties": True,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["orderedList"]},
                            "content": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/listItem"},
                            },
                        },
                        "required": ["type"],
                        "additionalProperties": True,
                    },
                ],
            },
            "listItem": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["listItem"]},
                    "content": {"type": "array", "items": {"$ref": "#/$defs/node"}},
                },
                "required": ["type"],
                "additionalProperties": True,
            },
        },
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "json_schema",
                "description": "Process the content using json schema validation",
                "parameters": circularRefSchema,
            },
        }
    ]

    _bedrock_tools_pt(tools=tools)


def test_bedrock_image_processor_content_type_fallback_url_extension():
    """
    Test that _post_call_image_processing falls back to URL extension
    when content-type is binary/octet-stream or application/octet-stream
    """
    import base64

    # Create mock response with binary/octet-stream content-type
    mock_response = MagicMock()
    mock_response.headers.get.return_value = "binary/octet-stream"

    # Create a simple PNG header (magic bytes)
    png_header = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a"
    png_content = png_header + b"\x00" * 100  # Add some padding
    mock_response.content = png_content

    # Test with .png URL
    image_url = "https://example.com/test-image.png"
    base64_bytes, content_type = BedrockImageProcessor._post_call_image_processing(
        mock_response, image_url
    )

    assert content_type == "image/png"
    assert base64_bytes == base64.b64encode(png_content).decode("utf-8")


def test_bedrock_image_processor_content_type_fallback_binary_detection():
    """
    Test that _post_call_image_processing falls back to binary content detection
    when content-type is missing and URL extension is not recognized
    """
    import base64

    # Create mock response with no content-type
    mock_response = MagicMock()
    mock_response.headers.get.return_value = None

    # Create a JPEG header (magic bytes)
    jpeg_header = b"\xff\xd8\xff"
    jpeg_content = jpeg_header + b"\x00" * 100  # Add some padding
    mock_response.content = jpeg_content

    # Test with URL without extension
    image_url = "https://example.com/test-image-without-extension"
    base64_bytes, content_type = BedrockImageProcessor._post_call_image_processing(
        mock_response, image_url
    )

    assert content_type == "image/jpeg"
    assert base64_bytes == base64.b64encode(jpeg_content).decode("utf-8")


def test_bedrock_image_processor_content_type_fallback_application_octet_stream():
    """
    Test that _post_call_image_processing handles application/octet-stream correctly
    """
    import base64

    # Create mock response with application/octet-stream content-type
    mock_response = MagicMock()
    mock_response.headers.get.return_value = "application/octet-stream"

    # Create a GIF header (magic bytes)
    gif_header = b"GIF8" + b"\x00" + b"a"
    gif_content = gif_header + b"\x00" * 100  # Add some padding
    mock_response.content = gif_content

    # Test with .gif URL
    image_url = "https://s3.amazonaws.com/bucket/image.gif"
    base64_bytes, content_type = BedrockImageProcessor._post_call_image_processing(
        mock_response, image_url
    )

    assert content_type == "image/gif"
    assert base64_bytes == base64.b64encode(gif_content).decode("utf-8")


def test_bedrock_image_processor_content_type_with_query_params():
    """
    Test that _post_call_image_processing correctly extracts extension from URL with query parameters
    """
    import base64

    # Create mock response with binary/octet-stream content-type
    mock_response = MagicMock()
    mock_response.headers.get.return_value = "binary/octet-stream"

    # Create a WebP header (magic bytes)
    webp_header = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP"
    webp_content = webp_header + b"\x00" * 100  # Add some padding
    mock_response.content = webp_content

    # Test with URL containing query parameters (common in S3 signed URLs)
    image_url = "https://s3.amazonaws.com/bucket/image.webp?AWSAccessKeyId=123&Expires=456&Signature=789"
    base64_bytes, content_type = BedrockImageProcessor._post_call_image_processing(
        mock_response, image_url
    )

    assert content_type == "image/webp"
    assert base64_bytes == base64.b64encode(webp_content).decode("utf-8")


def test_bedrock_image_processor_content_type_normal_header():
    """
    Test that _post_call_image_processing works normally when content-type is correctly set
    """
    import base64

    # Create mock response with correct content-type
    mock_response = MagicMock()
    mock_response.headers.get.return_value = "image/png"

    # Create a PNG header
    png_header = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a"
    png_content = png_header + b"\x00" * 100
    mock_response.content = png_content

    image_url = "https://example.com/test-image.png"
    base64_bytes, content_type = BedrockImageProcessor._post_call_image_processing(
        mock_response, image_url
    )

    assert content_type == "image/png"
    assert base64_bytes == base64.b64encode(png_content).decode("utf-8")


def test_bedrock_image_processor_content_type_fallback_failure():
    """
    Test that _post_call_image_processing raises ValueError when all fallback methods fail
    """
    # Create mock response with binary/octet-stream content-type
    mock_response = MagicMock()
    mock_response.headers.get.return_value = "binary/octet-stream"

    # Create content with unrecognizable image format
    mock_response.content = b"\x00" * 100

    # Test with URL without recognizable extension
    image_url = "https://example.com/unknown-file"

    with pytest.raises(ValueError) as excinfo:
        BedrockImageProcessor._post_call_image_processing(mock_response, image_url)

    assert "Unable to determine content type" in str(excinfo.value)


def test_bedrock_image_processor_content_type_jpeg_variants():
    """
    Test that _post_call_image_processing handles both .jpg and .jpeg extensions correctly
    """
    # Create mock response with binary/octet-stream
    mock_response = MagicMock()
    mock_response.headers.get.return_value = "binary/octet-stream"

    jpeg_header = b"\xff\xd8\xff"
    jpeg_content = jpeg_header + b"\x00" * 100
    mock_response.content = jpeg_content

    # Test with .jpg extension
    image_url_jpg = "https://example.com/photo.jpg"
    _, content_type_jpg = BedrockImageProcessor._post_call_image_processing(
        mock_response, image_url_jpg
    )
    assert content_type_jpg == "image/jpeg"

    # Test with .jpeg extension
    image_url_jpeg = "https://example.com/photo.jpeg"
    _, content_type_jpeg = BedrockImageProcessor._post_call_image_processing(
        mock_response, image_url_jpeg
    )
    assert content_type_jpeg == "image/jpeg"


def test_bedrock_image_processor_content_type_pdf_document():
    """
    Test that _post_call_image_processing handles PDF documents correctly
    when content-type is binary/octet-stream
    """
    import base64

    # Create mock response with binary/octet-stream content-type
    mock_response = MagicMock()
    mock_response.headers.get.return_value = "binary/octet-stream"

    # Create a PDF header (magic bytes: %PDF)
    pdf_header = b"%PDF-1.4"
    pdf_content = pdf_header + b"\x00" * 100
    mock_response.content = pdf_content

    # Test with .pdf URL
    pdf_url = "https://s3.amazonaws.com/bucket/document.pdf"
    base64_bytes, content_type = BedrockImageProcessor._post_call_image_processing(
        mock_response, pdf_url
    )

    assert content_type == "application/pdf"
    assert base64_bytes == base64.b64encode(pdf_content).decode("utf-8")


def test_bedrock_image_processor_content_type_document_formats():
    """
    Test that _post_call_image_processing handles various document formats
    """
    import base64

    # Create mock response
    mock_response = MagicMock()
    mock_response.headers.get.return_value = "application/octet-stream"
    mock_response.content = b"\x00" * 100

    # Test various document formats
    test_cases = [
        ("https://example.com/doc.pdf", "application/pdf"),
        ("https://example.com/sheet.csv", "text/csv"),
        (
            "https://example.com/doc.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        (
            "https://example.com/sheet.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        ("https://example.com/page.html", "text/html"),
        ("https://example.com/readme.txt", "text/plain"),
    ]

    for url, expected_mime in test_cases:
        _, content_type = BedrockImageProcessor._post_call_image_processing(
            mock_response, url
        )
        assert (
            content_type == expected_mime
        ), f"Expected {expected_mime} for {url}, got {content_type}"


def test_bedrock_image_processor_content_type_s3_pdf_with_query():
    """
    Test that _post_call_image_processing handles S3 PDF with query parameters
    """
    import base64

    # Create mock response
    mock_response = MagicMock()
    mock_response.headers.get.return_value = "binary/octet-stream"

    pdf_content = b"%PDF-1.4" + b"\x00" * 100
    mock_response.content = pdf_content

    # S3 signed URL with query parameters
    s3_url = "https://my-bucket.s3.us-east-1.amazonaws.com/documents/report.pdf?AWSAccessKeyId=AKIAIOSFODNN7EXAMPLE&Expires=1234567890&Signature=abcdef123456"

    base64_bytes, content_type = BedrockImageProcessor._post_call_image_processing(
        mock_response, s3_url
    )

    assert content_type == "application/pdf"
    assert base64_bytes == base64.b64encode(pdf_content).decode("utf-8")


def test_bedrock_tools_pt_empty_description():
    """
    Test that _bedrock_tools_pt handles empty string descriptions correctly.

    When a tool has an empty string description, Bedrock doesn't accept it,
    so the function should fall back to using the function name as the description.
    """
    from litellm.litellm_core_utils.prompt_templates.factory import _bedrock_tools_pt

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "",  # Empty string description
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    result = _bedrock_tools_pt(tools=tools)

    # Verify that the result is a list with one tool
    assert len(result) == 1

    # Verify that the description falls back to the function name
    tool_spec = result[0].get("toolSpec")
    assert tool_spec is not None
    assert tool_spec.get("name") == "get_weather"
    assert tool_spec.get("description") == "get_weather"


def test_bedrock_create_bedrock_block_deterministic_document_hash():
    """
    Test that _create_bedrock_block generates deterministic document names
    based on content hash. Same content should produce same hash.
    """
    import base64

    # Create PDF content
    pdf_content = b"%PDF-1.4\nSome PDF content here" + b"\x00" * 100
    base64_content = base64.b64encode(pdf_content).decode("utf-8")

    # Create two blocks with same content
    block1 = BedrockImageProcessor._create_bedrock_block(
        image_bytes=base64_content, mime_type="application/pdf", image_format="pdf"
    )
    block2 = BedrockImageProcessor._create_bedrock_block(
        image_bytes=base64_content, mime_type="application/pdf", image_format="pdf"
    )

    # Both should have the same document name
    assert block1.get("document") is not None
    assert block2.get("document") is not None
    assert block1["document"]["name"] == block2["document"]["name"]
    assert "DocumentPDFmessages_" in block1["document"]["name"]


def test_bedrock_create_bedrock_block_different_content_different_hash():
    """
    Test that different content produces different document hashes.
    """
    import base64

    # Create two different PDF contents
    pdf_content1 = b"%PDF-1.4\nFirst PDF content" + b"\x00" * 100
    pdf_content2 = b"%PDF-1.4\nSecond PDF content" + b"\x00" * 100

    base64_content1 = base64.b64encode(pdf_content1).decode("utf-8")
    base64_content2 = base64.b64encode(pdf_content2).decode("utf-8")

    # Create blocks
    block1 = BedrockImageProcessor._create_bedrock_block(
        image_bytes=base64_content1, mime_type="application/pdf", image_format="pdf"
    )
    block2 = BedrockImageProcessor._create_bedrock_block(
        image_bytes=base64_content2, mime_type="application/pdf", image_format="pdf"
    )

    # Should have different document names
    assert block1["document"]["name"] != block2["document"]["name"]


def test_bedrock_create_bedrock_block_normalized_base64():
    """
    Test that different base64 formatting (with/without whitespace)
    produces the same hash due to normalization.
    """
    import base64

    pdf_content = b"%PDF-1.4\nTest content" + b"\x00" * 100
    base64_content = base64.b64encode(pdf_content).decode("utf-8")

    # Create versions with different whitespace
    base64_with_newlines = "\n".join(
        [base64_content[i : i + 64] for i in range(0, len(base64_content), 64)]
    )
    base64_with_spaces = " ".join(
        [base64_content[i : i + 32] for i in range(0, len(base64_content), 32)]
    )

    # Create blocks
    block1 = BedrockImageProcessor._create_bedrock_block(
        image_bytes=base64_content, mime_type="application/pdf", image_format="pdf"
    )
    block2 = BedrockImageProcessor._create_bedrock_block(
        image_bytes=base64_with_newlines,
        mime_type="application/pdf",
        image_format="pdf",
    )
    block3 = BedrockImageProcessor._create_bedrock_block(
        image_bytes=base64_with_spaces,
        mime_type="application/pdf",
        image_format="pdf",
    )

    # All should have the same document name due to normalization
    assert block1["document"]["name"] == block2["document"]["name"]
    assert block1["document"]["name"] == block3["document"]["name"]


def test_bedrock_create_bedrock_block_large_file_sampling():
    """
    Test that files larger than 64KB use sampling correctly and
    different lengths produce different hashes.
    """
    import base64

    # Create two large files with same first 64KB but different total lengths
    first_64kb = b"%PDF-1.4\n" + b"A" * (64 * 1024)
    large_content1 = first_64kb + b"X" * 1024  # 65KB
    large_content2 = first_64kb + b"Y" * 2048  # 66KB

    base64_content1 = base64.b64encode(large_content1).decode("utf-8")
    base64_content2 = base64.b64encode(large_content2).decode("utf-8")

    # Create blocks
    block1 = BedrockImageProcessor._create_bedrock_block(
        image_bytes=base64_content1, mime_type="application/pdf", image_format="pdf"
    )
    block2 = BedrockImageProcessor._create_bedrock_block(
        image_bytes=base64_content2, mime_type="application/pdf", image_format="pdf"
    )

    # Should have different names because total length is different
    assert block1["document"]["name"] != block2["document"]["name"]


def test_bedrock_create_bedrock_block_very_large_file():
    """
    Test that very large files (>64KB) are handled correctly.
    """
    import base64

    # Create a large file (100KB)
    large_content = b"%PDF-1.4\n" + b"X" * (100 * 1024)
    base64_content = base64.b64encode(large_content).decode("utf-8")

    # Create block
    block = BedrockImageProcessor._create_bedrock_block(
        image_bytes=base64_content, mime_type="application/pdf", image_format="pdf"
    )

    # Should have a valid document name
    assert block.get("document") is not None
    assert "DocumentPDFmessages_" in block["document"]["name"]
    assert block["document"]["format"] == "pdf"


def test_bedrock_create_bedrock_block_image_type():
    """
    Test that image types still work correctly (no document name).
    """
    import base64

    # Create PNG content
    png_header = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a"
    png_content = png_header + b"\x00" * 100
    base64_content = base64.b64encode(png_content).decode("utf-8")

    # Create block
    block = BedrockImageProcessor._create_bedrock_block(
        image_bytes=base64_content, mime_type="image/png", image_format="png"
    )

    # Should be an image block, not a document block
    assert block.get("image") is not None
    assert block.get("document") is None
    assert block["image"]["format"] == "png"


def test_bedrock_create_bedrock_block_video_type():
    """
    Test that video types still work correctly (no document name).
    """
    import base64

    # Create MP4 content
    mp4_content = b"\x00\x00\x00\x20\x66\x74\x79\x70" + b"\x00" * 100
    base64_content = base64.b64encode(mp4_content).decode("utf-8")

    # Create block
    block = BedrockImageProcessor._create_bedrock_block(
        image_bytes=base64_content, mime_type="video/mp4", image_format="mp4"
    )

    # Should be a video block, not a document block
    assert block.get("video") is not None
    assert block.get("document") is None
    assert block["video"]["format"] == "mp4"


def test_bedrock_create_bedrock_block_document_name_format():
    """
    Test that document names follow the expected format:
    DocumentPDFmessages_{16_char_hash}_{format}
    """
    import base64
    import re

    pdf_content = b"%PDF-1.4\nTest content" + b"\x00" * 100
    base64_content = base64.b64encode(pdf_content).decode("utf-8")

    block = BedrockImageProcessor._create_bedrock_block(
        image_bytes=base64_content, mime_type="application/pdf", image_format="pdf"
    )

    document_name = block["document"]["name"]

    # Check format: DocumentPDFmessages_{16_hex_chars}_{format}
    pattern = r"^DocumentPDFmessages_[0-9a-f]{16}_pdf$"
    assert re.match(
        pattern, document_name
    ), f"Document name format mismatch: {document_name}"


def test_bedrock_create_bedrock_block_different_document_formats():
    """
    Test that different document formats (PDF, CSV, DOCX) are handled correctly.
    """
    import base64

    test_cases = [
        (b"%PDF-1.4\nContent", "application/pdf", "pdf"),
        (b"col1,col2\nval1,val2", "text/csv", "csv"),
        (
            b"PK\x03\x04",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "docx",
        ),
    ]

    for content, mime_type, format_type in test_cases:
        base64_content = base64.b64encode(content + b"\x00" * 100).decode("utf-8")
        block = BedrockImageProcessor._create_bedrock_block(
            image_bytes=base64_content, mime_type=mime_type, image_format=format_type
        )

        assert block.get("document") is not None
        assert f"DocumentPDFmessages_" in block["document"]["name"]
        assert block["document"]["name"].endswith(f"_{format_type}")
        assert block["document"]["format"] == format_type


def test_bedrock_nova_web_search_options_mapping():
    """
    Test that web_search_options is correctly mapped to Nova grounding.

    This follows the LiteLLM pattern for web search where:
    - Vertex AI maps web_search_options to {"googleSearch": {}}
    - Anthropic maps web_search_options to {"type": "web_search_20250305", ...}
    - Nova should map web_search_options to {"systemTool": {"name": "nova_grounding"}}
    """
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

    config = AmazonConverseConfig()

    # Test basic mapping for Nova model
    result = config._map_web_search_options({}, "amazon.nova-pro-v1:0")

    assert result is not None
    system_tool = result.get("systemTool")
    assert system_tool is not None
    assert system_tool["name"] == "nova_grounding"

    # Test with search_context_size (should be ignored for Nova)
    result2 = config._map_web_search_options(
        {"search_context_size": "high"}, "us.amazon.nova-premier-v1:0"
    )

    assert result2 is not None
    system_tool2 = result2.get("systemTool")
    assert system_tool2 is not None
    assert system_tool2["name"] == "nova_grounding"
    # Nova doesn't support search_context_size, so it's just ignored


def test_bedrock_tools_pt_does_not_handle_system_tool():
    """
    Verify that _bedrock_tools_pt does NOT handle system_tool format.

    System tools (nova_grounding) should be added via web_search_options,
    not via the tools parameter directly.
    """

    from litellm.litellm_core_utils.prompt_templates.factory import _bedrock_tools_pt

    # Regular function tools should still work
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    result = _bedrock_tools_pt(tools=tools)

    assert len(result) == 1
    tool_spec = result[0].get("toolSpec")
    assert tool_spec is not None
    assert tool_spec["name"] == "get_weather"


def test_convert_to_anthropic_tool_result_image_with_cache_control():
    """
    Test that cache_control is properly applied to image content in tool results.
    This tests the functionality added in the uncommitted changes where
    add_cache_control_to_content is called for image_url content types.
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_anthropic_tool_result,
    )

    # Test with base64 image data URI
    message = {
        "role": "tool",
        "tool_call_id": "call_test_123",
        "content": [
            {
                "type": "text",
                "text": "Here is the image you requested:",
            },
            {
                "type": "image_url",
                "image_url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQ",
                "cache_control": {"type": "ephemeral"},
            },
        ],
    }

    result = convert_to_anthropic_tool_result(message)

    # Verify the result structure
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "call_test_123"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 2

    # Verify text content
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"] == "Here is the image you requested:"

    # Verify image content with cache_control
    assert result["content"][1]["type"] == "image"
    assert result["content"][1]["source"]["type"] == "base64"
    assert result["content"][1]["source"]["media_type"] == "image/jpeg"
    assert "cache_control" in result["content"][1]
    assert result["content"][1]["cache_control"]["type"] == "ephemeral"


def test_convert_to_anthropic_tool_result_image_without_cache_control():
    """
    Test that images without cache_control in tool results work correctly.
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_anthropic_tool_result,
    )

    message = {
        "role": "tool",
        "tool_call_id": "call_test_456",
        "content": [
            {
                "type": "image_url",
                "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA",
            },
        ],
    }

    result = convert_to_anthropic_tool_result(message)

    # Verify the result structure
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "call_test_456"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 1

    # Verify image content without cache_control (cache_control will be None if not set)
    assert result["content"][0]["type"] == "image"
    assert result["content"][0]["source"]["type"] == "base64"
    assert result["content"][0]["source"]["media_type"] == "image/png"
    assert result["content"][0].get("cache_control") is None


def test_convert_to_anthropic_tool_result_mixed_content_with_cache_control():
    """
    Test tool results with mixed content types (text and image) where only some have cache_control.
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_anthropic_tool_result,
    )

    message = {
        "role": "tool",
        "tool_call_id": "call_test_789",
        "content": [
            {
                "type": "text",
                "text": "First image:",
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "image_url",
                "image_url": "data:image/jpeg;base64,/9j/4AAQSkZJRg",
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": "Second image (no cache):",
            },
            {
                "type": "image_url",
                "image_url": "data:image/png;base64,iVBORw0KGgo",
            },
        ],
    }

    result = convert_to_anthropic_tool_result(message)

    assert result["type"] == "tool_result"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 4

    # First text with cache_control
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["cache_control"]["type"] == "ephemeral"

    # First image with cache_control
    assert result["content"][1]["type"] == "image"
    assert result["content"][1]["cache_control"]["type"] == "ephemeral"

    # Second text without cache_control (cache_control will be None if not set)
    assert result["content"][2]["type"] == "text"
    assert result["content"][2].get("cache_control") is None

    # Second image without cache_control (cache_control will be None if not set)
    assert result["content"][3]["type"] == "image"
    assert result["content"][3].get("cache_control") is None


def test_convert_to_anthropic_tool_result_image_url_as_http():
    """
    Test that HTTP/HTTPS URLs with cache_control are handled correctly.
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_anthropic_tool_result,
    )

    message = {
        "role": "tool",
        "tool_call_id": "call_http_001",
        "content": [
            {
                "type": "image_url",
                "image_url": "https://example.com/image.jpg",
                "cache_control": {"type": "ephemeral"},
            },
        ],
    }

    result = convert_to_anthropic_tool_result(message)

    # Verify image is passed as URL reference with cache_control
    assert result["content"][0]["type"] == "image"
    assert result["content"][0]["source"]["type"] == "url"
    assert result["content"][0]["source"]["url"] == "https://example.com/image.jpg"
    assert result["content"][0]["cache_control"]["type"] == "ephemeral"


def test_anthropic_messages_pt_server_tool_use_passthrough():
    """
    Test that anthropic_messages_pt passes through server_tool_use and
    tool_search_tool_result blocks in assistant message content.

    These are Anthropic-native content types used for tool search functionality
    that need to be preserved when reconstructing multi-turn conversations.

    Fixes: https://github.com/BerriAI/litellm/issues/XXXXX
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        anthropic_messages_pt,
    )

    messages = [
        {"role": "user", "content": "I need help with time information."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_01ABC123",
                    "name": "tool_search_tool_regex",
                    "input": {"query": ".*time.*"},
                },
                {
                    "type": "tool_search_tool_result",
                    "tool_use_id": "srvtoolu_01ABC123",
                    "content": {
                        "type": "tool_search_tool_search_result",
                        "tool_references": [
                            {"type": "tool_reference", "tool_name": "get_time"}
                        ],
                    },
                },
                {"type": "text", "text": "I found the time tool. How can I help you?"},
            ],
        },
        {"role": "user", "content": "What's the time in New York?"},
    ]

    result = anthropic_messages_pt(
        messages=messages,
        model="claude-sonnet-4-5-20250929",
        llm_provider="anthropic",
    )

    # Verify we have 3 messages (user, assistant, user)
    assert len(result) == 3

    # Verify the assistant message content
    assistant_msg = result[1]
    assert assistant_msg["role"] == "assistant"
    assert isinstance(assistant_msg["content"], list)

    # Find the different content block types
    content_types = [block.get("type") for block in assistant_msg["content"]]

    # Verify server_tool_use block is preserved
    assert "server_tool_use" in content_types
    server_tool_use_block = next(
        b for b in assistant_msg["content"] if b.get("type") == "server_tool_use"
    )
    assert server_tool_use_block["id"] == "srvtoolu_01ABC123"
    assert server_tool_use_block["name"] == "tool_search_tool_regex"
    assert server_tool_use_block["input"] == {"query": ".*time.*"}

    # Verify tool_search_tool_result block is preserved
    assert "tool_search_tool_result" in content_types
    tool_result_block = next(
        b
        for b in assistant_msg["content"]
        if b.get("type") == "tool_search_tool_result"
    )
    assert tool_result_block["tool_use_id"] == "srvtoolu_01ABC123"
    assert tool_result_block["content"]["type"] == "tool_search_tool_search_result"
    assert tool_result_block["content"]["tool_references"][0]["tool_name"] == "get_time"

    # Verify text block is also preserved
    assert "text" in content_types
    text_block = next(b for b in assistant_msg["content"] if b.get("type") == "text")
    assert text_block["text"] == "I found the time tool. How can I help you?"


def test_bedrock_tools_unpack_defs_no_oom_with_nested_refs():
    """
    Regression test for issue #19098: unpack_defs() causes OOM with nested tool schemas.

    The old implementation had a "flatten defs" loop that would pre-expand each def
    using unpack_defs(), but since defs often reference each other, each subsequent
    call would copy already-expanded content, causing exponential memory growth.

    This test creates a schema with multiple nested $defs that reference each other
    to verify the fix prevents memory explosion while still correctly resolving refs.
    """
    import sys
    import copy

    from litellm.litellm_core_utils.prompt_templates.factory import _bedrock_tools_pt

    # Schema with multiple nested $defs that reference each other
    # This pattern would cause OOM with the old "flatten defs" loop
    complex_nested_schema = {
        "type": "object",
        "properties": {
            "query": {"$ref": "#/$defs/Expression"},
        },
        "$defs": {
            "Expression": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["and", "or", "not", "comparison"],
                    },
                    "left": {"$ref": "#/$defs/Operand"},
                    "right": {"$ref": "#/$defs/Operand"},
                    "operator": {"$ref": "#/$defs/Operator"},
                },
            },
            "Operand": {
                "type": "object",
                "anyOf": [
                    {"$ref": "#/$defs/Literal"},
                    {"$ref": "#/$defs/FieldRef"},
                    {
                        "$ref": "#/$defs/Expression"
                    },  # Circular: Operand -> Expression -> Operand
                ],
            },
            "Literal": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "const": "literal"},
                    "value": {"$ref": "#/$defs/LiteralValue"},
                },
            },
            "LiteralValue": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "null"},
                ],
            },
            "FieldRef": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "const": "field"},
                    "name": {"type": "string"},
                    "table": {"$ref": "#/$defs/TableRef"},
                },
            },
            "TableRef": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "alias": {"type": "string"},
                },
            },
            "Operator": {
                "type": "string",
                "enum": ["=", "!=", "<", ">", "<=", ">=", "LIKE", "IN"],
            },
        },
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_query",
                "description": "Execute a query with complex expressions",
                "parameters": complex_nested_schema,
            },
        }
    ]

    # Measure initial size
    def get_size(obj, seen=None):
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum([get_size(v, seen) for v in obj.values()])
            size += sum([get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([get_size(i, seen) for i in obj])
        return size

    initial_size = get_size(tools)

    # Process through _bedrock_tools_pt - this should complete without OOM
    tools_copy = copy.deepcopy(tools)
    result = _bedrock_tools_pt(tools=tools_copy)

    final_size = get_size(result)

    # The expansion factor should be reasonable (< 100x), not exponential (35000x as in #19098)
    expansion_factor = final_size / initial_size
    assert expansion_factor < 100, (
        f"Memory expansion factor {expansion_factor:.1f}x is too high. "
        f"Initial: {initial_size} bytes, Final: {final_size} bytes"
    )

    # Verify the result is valid Bedrock tools format
    assert isinstance(result, list)
    assert len(result) == 1
    assert "toolSpec" in result[0]
    assert result[0]["toolSpec"]["name"] == "execute_query"

    # Verify $defs have been removed (Bedrock doesn't support them)
    tool_schema = result[0]["toolSpec"].get("inputSchema", {}).get("json", {})
    assert "$defs" not in tool_schema, "$defs should be removed after expansion"


def test_anthropic_messages_pt_file_block_preserves_cache_control():
    """
    Test that cache_control on file-type content blocks is preserved
    when translating to Anthropic message format.
    Regression test for https://github.com/BerriAI/litellm/issues/23873
    """

    pdf_b64 = base64.b64encode(b"%PDF-1.4 fake pdf content").decode()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "filename": "document.pdf",
                        "file_data": f"data:application/pdf;base64,{pdf_b64}",
                    },
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": "Summarize this document.",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        }
    ]

    result = anthropic_messages_pt(
        messages=messages,
        model="claude-sonnet-4-20250514",
        llm_provider="anthropic",
    )

    assert len(result) == 1
    content_blocks = result[0]["content"]
    assert len(content_blocks) == 2

    file_block = content_blocks[0]
    assert file_block["type"] == "document"
    assert (
        "cache_control" in file_block
    ), "cache_control should be preserved on file/document content blocks"
    assert file_block["cache_control"]["type"] == "ephemeral"

    text_block = content_blocks[1]
    assert text_block["type"] == "text"
    assert "cache_control" in text_block
    assert text_block["cache_control"]["type"] == "ephemeral"


def test_anthropic_messages_pt_file_block_without_cache_control():
    """
    Test that file blocks without cache_control still work correctly.
    """
    import base64

    pdf_b64 = base64.b64encode(b"%PDF-1.4 fake").decode()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "filename": "doc.pdf",
                        "file_data": f"data:application/pdf;base64,{pdf_b64}",
                    },
                },
            ],
        }
    ]

    result = anthropic_messages_pt(
        messages=messages,
        model="claude-sonnet-4-20250514",
        llm_provider="anthropic",
    )

    assert len(result) == 1
    file_block = result[0]["content"][0]
    assert file_block["type"] == "document"
    assert "cache_control" not in file_block


# ── _convert_to_bedrock_tool_call_invoke tests ──


def test_bedrock_tool_call_invoke_normal_single_tool():
    """Normal single tool call with valid JSON arguments."""
    tool_calls = [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Boston, MA"}',
            },
        }
    ]
    result = _convert_to_bedrock_tool_call_invoke(tool_calls)
    assert len(result) == 1
    assert result[0]["toolUse"]["toolUseId"] == "call_abc123"
    assert result[0]["toolUse"]["name"] == "get_weather"
    assert result[0]["toolUse"]["input"] == {"location": "Boston, MA"}


def test_bedrock_tool_call_invoke_empty_arguments():
    """Tool call with empty arguments produces an empty dict input."""
    tool_calls = [
        {
            "id": "call_empty",
            "type": "function",
            "function": {"name": "do_something", "arguments": ""},
        }
    ]
    result = _convert_to_bedrock_tool_call_invoke(tool_calls)
    assert len(result) == 1
    assert result[0]["toolUse"]["input"] == {}


def test_bedrock_tool_call_invoke_concatenated_json():
    """
    Tool call whose arguments contain multiple concatenated JSON objects
    (the bug from issue #20543) is split into separate Bedrock toolUse blocks.

    Bedrock Claude Sonnet 4.5 sometimes returns multiple tool call arguments
    concatenated in a single string like:
        '{"command":["curl",...]}{"command":["curl",...]}{"command":["curl",...]}'
    """
    tool_calls = [
        {
            "id": "tooluse_L7I3TewYAUhoheJZQEuwVN",
            "type": "function",
            "function": {
                "name": "shell",
                "arguments": (
                    '{"command": ["curl", "-i", "http://localhost:9009", "-m", "10"]}'
                    '{"command": ["curl", "-i", "http://localhost:9009/robots.txt", "-m", "5"]}'
                    '{"command": ["curl", "-i", "http://localhost:9009/sitemap.xml", "-m", "5"]}'
                ),
            },
        }
    ]
    result = _convert_to_bedrock_tool_call_invoke(tool_calls)

    # Should produce 3 separate toolUse blocks
    assert len(result) == 3

    # First block keeps original tool id
    assert result[0]["toolUse"]["toolUseId"] == "tooluse_L7I3TewYAUhoheJZQEuwVN"
    assert result[0]["toolUse"]["name"] == "shell"
    assert result[0]["toolUse"]["input"] == {
        "command": ["curl", "-i", "http://localhost:9009", "-m", "10"]
    }

    # Subsequent blocks get suffixed ids
    assert result[1]["toolUse"]["toolUseId"] == "tooluse_L7I3TewYAUhoheJZQEuwVN_1"
    assert result[1]["toolUse"]["name"] == "shell"
    assert result[1]["toolUse"]["input"] == {
        "command": ["curl", "-i", "http://localhost:9009/robots.txt", "-m", "5"]
    }

    assert result[2]["toolUse"]["toolUseId"] == "tooluse_L7I3TewYAUhoheJZQEuwVN_2"
    assert result[2]["toolUse"]["name"] == "shell"
    assert result[2]["toolUse"]["input"] == {
        "command": ["curl", "-i", "http://localhost:9009/sitemap.xml", "-m", "5"]
    }


def test_bedrock_tool_call_invoke_concatenated_json_with_cache_control():
    """
    When a tool call has cache_control AND concatenated JSON arguments,
    the cachePoint block is appended after the last split block.
    """
    tool_calls = [
        {
            "id": "call_cached",
            "type": "function",
            "cache_control": {"type": "default"},
            "function": {
                "name": "shell",
                "arguments": '{"a": 1}{"b": 2}',
            },
        }
    ]
    result = _convert_to_bedrock_tool_call_invoke(tool_calls)

    # 2 toolUse blocks + 1 cachePoint block
    assert len(result) == 3
    assert "toolUse" in result[0]
    assert "toolUse" in result[1]
    assert "cachePoint" in result[2]


def test_bedrock_tool_call_invoke_non_dict_arguments():
    """Arguments that parse to a non-dict (e.g. '""') produce empty dict input."""
    tool_calls = [
        {
            "id": "call_non_dict",
            "type": "function",
            "function": {"name": "tool", "arguments": '""'},
        }
    ]
    result = _convert_to_bedrock_tool_call_invoke(tool_calls)
    assert len(result) == 1
    assert result[0]["toolUse"]["input"] == {}


def test_bedrock_tool_call_invoke_multiple_normal_tools():
    """Multiple separate tool calls (normal parallel calling) work correctly."""
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "NYC"}',
            },
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "LA"}',
            },
        },
    ]
    result = _convert_to_bedrock_tool_call_invoke(tool_calls)
    assert len(result) == 2
    assert result[0]["toolUse"]["toolUseId"] == "call_1"
    assert result[1]["toolUse"]["toolUseId"] == "call_2"


# ========================================================================
# Tool result deduplication tests (Case D in sanitize_messages_for_tool_calling)
# ========================================================================


def test_sanitize_messages_deduplicates_tool_results():
    """
    Anthropic requires exactly one tool_result per tool_use. When conversation
    history (e.g. from session resume) contains duplicate tool result messages
    with the same tool_call_id, sanitize_messages_for_tool_calling should keep
    only the last occurrence.

    Without this fix, Anthropic rejects with:
        each tool_use must have a single result. Found multiple tool_result
        blocks with id: <id>
    """
    original = litellm.modify_params
    litellm.modify_params = True
    try:
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    }
                ],
            },
            # First tool result (stale/duplicate)
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "Partial result...",
            },
            # Second tool result (final/complete — should be kept)
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": '{"temperature": 72, "condition": "sunny"}',
            },
        ]

        result = sanitize_messages_for_tool_calling(messages)

        # Count tool messages with this ID — should be exactly 1
        tool_results = [
            m
            for m in result
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_abc123"
        ]
        assert len(tool_results) == 1
        # Should keep the LAST occurrence (most complete)
        assert tool_results[0]["content"] == '{"temperature": 72, "condition": "sunny"}'
    finally:
        litellm.modify_params = original


def test_sanitize_messages_preserves_unique_tool_results():
    """
    When each tool_call_id has exactly one tool_result, no deduplication should
    occur. Messages should pass through unchanged.
    """
    original = litellm.modify_params
    litellm.modify_params = True
    try:
        messages = [
            {"role": "user", "content": "Get weather for two cities"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "LA"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "72F"},
            {"role": "tool", "tool_call_id": "call_2", "content": "85F"},
        ]

        result = sanitize_messages_for_tool_calling(messages)

        tool_results = [m for m in result if m.get("role") == "tool"]
        assert len(tool_results) == 2
        assert tool_results[0]["tool_call_id"] == "call_1"
        assert tool_results[0]["content"] == "72F"
        assert tool_results[1]["tool_call_id"] == "call_2"
        assert tool_results[1]["content"] == "85F"
    finally:
        litellm.modify_params = original


def test_sanitize_messages_dedup_disabled_when_modify_params_false():
    """
    When litellm.modify_params is False, messages should be returned as-is
    even if they contain duplicate tool results.
    """
    original = litellm.modify_params
    litellm.modify_params = False
    try:
        messages = [
            {"role": "user", "content": "Test"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_dup",
                        "type": "function",
                        "function": {"name": "test", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_dup", "content": "first"},
            {"role": "tool", "tool_call_id": "call_dup", "content": "second"},
        ]

        result = sanitize_messages_for_tool_calling(messages)

        # Should be unchanged — no sanitization when modify_params=False
        assert result == messages
    finally:
        litellm.modify_params = original


def test_sanitize_messages_dedup_scoped_per_turn_preserves_cross_turn():
    """
    When the same tool_call_id appears in two different assistant turns
    (separated by a user message), both tool results must be preserved.
    Deduplication should only apply within a single contiguous tool-result
    block, not globally across the conversation.

    Without per-turn scoping this would incorrectly drop the first tool result,
    leaving the first assistant message without its required result (which
    Anthropic would reject).
    """
    original = litellm.modify_params
    litellm.modify_params = True
    try:
        messages = [
            {"role": "user", "content": "First question"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_X",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q": "a"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_X", "content": "result_turn_1"},
            {"role": "user", "content": "Second question"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_X",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q": "b"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_X", "content": "result_turn_2"},
        ]

        result = sanitize_messages_for_tool_calling(messages)

        # Both tool results must survive — one per turn
        tool_results = [
            m
            for m in result
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_X"
        ]
        assert len(tool_results) == 2, (
            f"Expected 2 tool results (one per turn), got {len(tool_results)}. "
            "Dedup may be global instead of per-turn scoped."
        )
        assert tool_results[0]["content"] == "result_turn_1"
        assert tool_results[1]["content"] == "result_turn_2"
    finally:
        litellm.modify_params = original


def test_sanitize_messages_combined_case_a_and_case_d():
    """
    Combined Case A + Case D: an assistant message has two tool_calls —
    one with a missing result (Case A should inject a dummy) and one with
    duplicate results (Case D should deduplicate to keep only the last).

    This validates that both sanitization passes compose correctly without
    interfering with each other.
    """
    original = litellm.modify_params
    litellm.modify_params = True
    try:
        messages = [
            {"role": "user", "content": "Do two things"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_missing",
                        "type": "function",
                        "function": {"name": "tool_a", "arguments": "{}"},
                    },
                    {
                        "id": "call_duped",
                        "type": "function",
                        "function": {"name": "tool_b", "arguments": '{"q": "x"}'},
                    },
                ],
            },
            # No result for call_missing — Case A should inject a dummy
            # Duplicate results for call_duped — Case D should keep last
            {"role": "tool", "tool_call_id": "call_duped", "content": "stale_result"},
            {"role": "tool", "tool_call_id": "call_duped", "content": "fresh_result"},
            {"role": "user", "content": "Now summarize"},
        ]

        result = sanitize_messages_for_tool_calling(messages)

        # Collect tool results from the output
        tool_results = [m for m in result if m.get("role") in ("tool", "function")]

        # Case A: call_missing should have a dummy result injected
        missing_results = [
            m for m in tool_results if m.get("tool_call_id") == "call_missing"
        ]
        assert (
            len(missing_results) == 1
        ), f"Expected 1 dummy result for call_missing (Case A), got {len(missing_results)}"

        # Case D: call_duped should have exactly 1 result (the fresh one)
        duped_results = [
            m for m in tool_results if m.get("tool_call_id") == "call_duped"
        ]
        assert (
            len(duped_results) == 1
        ), f"Expected 1 result for call_duped after dedup (Case D), got {len(duped_results)}"
        assert (
            duped_results[0]["content"] == "fresh_result"
        ), f"Expected last-wins 'fresh_result', got '{duped_results[0]['content']}'"

        # Verify tool results immediately follow the assistant message
        asst_idx = next(i for i, m in enumerate(result) if m.get("role") == "assistant")
        tool_msgs_after_asst = [
            m for m in result[asst_idx + 1 :] if m.get("role") in ("tool", "function")
        ]
        assert (
            len(tool_msgs_after_asst) == 2
        ), f"Expected 2 tool results after assistant, got {len(tool_msgs_after_asst)}"
        # Both tool_call_ids should be present (order may vary)
        tool_ids = {m["tool_call_id"] for m in tool_msgs_after_asst}
        assert tool_ids == {
            "call_missing",
            "call_duped",
        }, f"Expected tool_call_ids {{call_missing, call_duped}}, got {tool_ids}"
    finally:
        litellm.modify_params = original


def test_anthropic_messages_pt_file_block_preserves_cache_control():
    """
    Test that cache_control is preserved on file-type content blocks
    when translated to Anthropic document params.
    Regression test for https://github.com/BerriAI/litellm/issues/23873
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        anthropic_messages_pt,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "filename": "doc.pdf",
                        "file_data": "data:application/pdf;base64,JVBERi0xLjQ=",
                    },
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": "Summarize this document.",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        }
    ]

    result = anthropic_messages_pt(
        messages, model="claude-sonnet-4-20250514", llm_provider="anthropic"
    )

    content_blocks = result[0]["content"]
    assert len(content_blocks) == 2

    # Document block (from file) should preserve cache_control
    doc_block = content_blocks[0]
    assert doc_block["type"] == "document"
    assert (
        "cache_control" in doc_block
    ), "cache_control was dropped from file/document block"
    assert doc_block["cache_control"]["type"] == "ephemeral"

    # Text block should also preserve cache_control
    text_block = content_blocks[1]
    assert text_block["type"] == "text"
    assert "cache_control" in text_block
    assert text_block["cache_control"]["type"] == "ephemeral"


def test_add_cache_point_tool_block_passes_ttl_for_claude_4_5():
    """
    Tools with cache_control ttl should preserve the ttl in the cachePoint
    block for Claude 4.5+ models on Bedrock, matching the behavior of system
    block cache_control.

    Without this fix, tool cachePoint is always {"type": "default"} (5m),
    while system blocks can have ttl="1h", violating Bedrock's non-increasing
    TTL ordering constraint (tools -> system -> messages).

    Ref: https://github.com/BerriAI/litellm/issues/XXXXX
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        add_cache_point_tool_block,
    )

    tool_with_1h = {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }

    # Claude 4.5 model: ttl should be preserved
    result = add_cache_point_tool_block(
        tool_with_1h, model="us.anthropic.claude-sonnet-4-5-20250514-v1:0"
    )
    assert result is not None
    assert result["cachePoint"]["type"] == "default"
    assert result["cachePoint"]["ttl"] == "1h"

    # Claude 4.5 model with 5m ttl: also preserved
    tool_with_5m = {
        "cache_control": {"type": "ephemeral", "ttl": "5m"},
    }
    result_5m = add_cache_point_tool_block(
        tool_with_5m, model="us.anthropic.claude-sonnet-4-5-20250514-v1:0"
    )
    assert result_5m is not None
    assert result_5m["cachePoint"]["ttl"] == "5m"

    # Older model: ttl should be stripped
    result_old = add_cache_point_tool_block(
        tool_with_1h, model="anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    assert result_old is not None
    assert result_old["cachePoint"]["type"] == "default"
    assert "ttl" not in result_old["cachePoint"]

    # No model provided: ttl should be stripped (safe default)
    result_no_model = add_cache_point_tool_block(tool_with_1h, model=None)
    assert result_no_model is not None
    assert "ttl" not in result_no_model["cachePoint"]

    # No cache_control: returns None (unchanged behavior)
    tool_no_cache = {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }
    assert add_cache_point_tool_block(tool_no_cache) is None

    # cache_control without ttl: returns default cachePoint (unchanged behavior)
    tool_no_ttl = {"cache_control": {"type": "ephemeral"}}
    result_no_ttl = add_cache_point_tool_block(
        tool_no_ttl, model="us.anthropic.claude-sonnet-4-5-20250514-v1:0"
    )
    assert result_no_ttl is not None
    assert result_no_ttl["cachePoint"]["type"] == "default"
    assert "ttl" not in result_no_ttl["cachePoint"]


def test_bedrock_tools_pt_passes_ttl_for_claude_4_5():
    """
    End-to-end: _bedrock_tools_pt should produce cachePoint blocks with ttl
    for Claude 4.5+ models when tools have cache_control with ttl.
    """
    from litellm.litellm_core_utils.prompt_templates.factory import _bedrock_tools_pt

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]

    # Claude 4.5: cachePoint should have ttl
    result = _bedrock_tools_pt(
        tools, model="us.anthropic.claude-sonnet-4-5-20250514-v1:0"
    )
    cache_blocks = [b for b in result if "cachePoint" in b]
    assert len(cache_blocks) == 1
    assert cache_blocks[0]["cachePoint"]["ttl"] == "1h"

    # Older model: cachePoint should not have ttl
    result_old = _bedrock_tools_pt(
        tools, model="anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    cache_blocks_old = [b for b in result_old if "cachePoint" in b]
    assert len(cache_blocks_old) == 1
    assert "ttl" not in cache_blocks_old[0]["cachePoint"]


def test_convert_to_anthropic_tool_result_openai_file_pdf_becomes_document():
    """
    OpenAI `{type: "file", file: {file_data: "data:application/pdf;..."}}` inside
    a tool-message content list should translate to an Anthropic document block
    inside the tool_result content. Reuses anthropic_process_openai_file_message,
    which already handles this for user messages.
    """
    pdf_b64 = "JVBERi0xLjQKJeLjz9MK"
    message = {
        "tool_call_id": "toolu_pdf_1",
        "role": "tool",
        "name": "fetch_document",
        "content": [
            {
                "type": "file",
                "file": {
                    "file_data": f"data:application/pdf;base64,{pdf_b64}",
                    "filename": "summary.pdf",
                },
            },
        ],
    }

    result = _convert_to_bedrock_tool_call_result(message)

    tool_result = result["toolResult"]
    assert len(tool_result["content"]) == 1
    assert "document" in tool_result["content"][0]
    assert tool_result["content"][0]["document"]["format"] == "pdf"
    assert tool_result["content"][0]["document"]["source"]["bytes"] == pdf_b64


def test_bedrock_converse_messages_pt_document_various_formats():
    """Test that various document media types produce the correct format value."""
    test_cases = [
        ("application/pdf", "pdf"),
        ("text/csv", "csv"),
        ("text/html", "html"),
        ("text/plain", "txt"),
        ("text/markdown", "md"),
        (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "docx",
        ),
    ]

    for media_type, expected_format in test_cases:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": "dGVzdA==",
                        },
                    },
                ],
            }
        ]

        result = _bedrock_converse_messages_pt(
            messages, "anthropic.claude-sonnet-4-6", "bedrock"
        )

        doc_block = result[0]["content"][0]
        assert doc_block["document"]["format"] == expected_format, (
            f"Expected format '{expected_format}' for media_type '{media_type}', "
            f"got '{doc_block['document']['format']}'"
        )


def test_bedrock_converse_messages_pt_document_deterministic_name():
    """Test that the same document data always produces the same name."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "dGVzdA==",
                    },
                },
            ],
        }
    ]

    result1 = _bedrock_converse_messages_pt(
        messages, "anthropic.claude-sonnet-4-6", "bedrock"
    )
    result2 = _bedrock_converse_messages_pt(
        messages, "anthropic.claude-sonnet-4-6", "bedrock"
    )

    name1 = result1[0]["content"][0]["document"]["name"]
    name2 = result2[0]["content"][0]["document"]["name"]
    assert name1 == name2


def test_bedrock_converse_messages_pt_document_rejects_url_source():
    """Test that a URL-type document source raises a clear error instead of KeyError."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/doc.pdf",
                    },
                },
            ],
        }
    ]

    with pytest.raises(ValueError, match="only supports base64-encoded"):
        _bedrock_converse_messages_pt(
            messages, "anthropic.claude-sonnet-4-6", "bedrock"
        )
