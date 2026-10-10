import asyncio, base64, importlib, uuid
import json
import logging
import os
import re
from typing import Final, List
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import (
    BEDROCK_DOCUMENT_PLACEHOLDER_TEXT,
    BedrockConverseMessagesProcessor,
    BedrockImageProcessor,
    _bedrock_converse_messages_pt,
    _bedrock_tools_pt,
    _rename_duplicate_bedrock_document_names,
    _convert_to_bedrock_tool_call_invoke,
    _sanitize_anthropic_tool_use_id,
    _convert_to_bedrock_tool_call_result,
    anthropic_messages_pt,
    convert_to_anthropic_tool_result,
    convert_to_gemini_tool_call_result,
    encode_tool_call_id_with_signature,
    function_call_prompt,
    get_thought_signature_from_tool,
    get_tool_calls_from_response,
    make_valid_bedrock_tool_name,
    ollama_pt,
    parse_mime_type,
    sanitize_messages_for_tool_calling,
    THOUGHT_SIGNATURE_SEPARATOR,
)
from litellm.types.llms.openai import ChatCompletionToolMessage
from litellm.utils import (
    _invalidate_model_cost_lowercase_map,
    function_setup,
    Rules,
    validate_and_fix_openai_messages,
)
from datetime import datetime
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome
from litellm.litellm_core_utils.prompt_templates.factory import (
    anthropic_pt,
    claude_2_1_pt,
    convert_to_anthropic_image_obj,
    convert_to_anthropic_tool_invoke,
    convert_url_to_base64,
    create_anthropic_image_param,
    has_tool_with_name,
    llama_2_chat_pt,
    prompt_factory,
)
from litellm.litellm_core_utils.prompt_templates.common_utils import get_completion_messages
from litellm.llms.vertex_ai.gemini.transformation import gemini_convert_messages_with_history
from litellm.types.llms.openai import AllMessageValues


def test_function_call_prompt_preserves_append_failure_for_non_string_content() -> None:
    messages = [{"role": "system", "content": None}]

    with pytest.raises(AttributeError):
        function_call_prompt(messages, [])


def test_function_call_prompt_lets_the_model_answer_after_a_function_result() -> None:
    messages: Final[list[dict[str, object]]] = [{"role": "system", "content": "Be terse."}]

    prompted: Final = function_call_prompt(messages, [{"name": "get_weather"}])

    system: Final = str(prompted[0]["content"])
    assert "JSON OUTPUT ONLY" not in system
    assert "reply to the user in plain text instead of calling a function again" in system
    assert "{'name': 'get_weather'}" in system


@pytest.mark.parametrize(
    ("thought_signature", "expected"),
    [
        ("encoded-signature", "call_123__thought__encoded-signature"),
        (None, "call_123"),
        ("", "call_123"),
    ],
)
def test_encode_tool_call_id_with_signature(thought_signature, expected):
    assert encode_tool_call_id_with_signature("call_123", thought_signature) == expected


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ({"provider_specific_fields": {"thought_signature": "tool-signature"}}, "tool-signature"),
        (
            {"function": {"provider_specific_fields": {"thought_signature": "function-signature"}}},
            "function-signature",
        ),
        (
            {"id": encode_tool_call_id_with_signature("call_123", "embedded-signature")},
            "embedded-signature",
        ),
        ({}, None),
    ],
)
def test_get_thought_signature_from_tool(tool, expected):
    assert get_thought_signature_from_tool(tool) == expected


@pytest.mark.parametrize(
    ("base64_data", "expected"),
    [
        ("data:image/png;base64,encoded-image", "image/png"),
        ("data:application/pdf;base64,encoded-document", "application/pdf"),
        ("not-a-data-url", None),
    ],
)
def test_parse_mime_type(base64_data, expected):
    assert parse_mime_type(base64_data) == expected


@pytest.mark.parametrize(
    ("message", "expected_error"),
    [
        ({"type": "file"}, "missing the required 'file' field"),
        ({"type": "file", "file": {}}, "file_data and file_id cannot both be None"),
    ],
)
def test_process_file_message_rejects_missing_file_data(message, expected_error):
    with pytest.raises(litellm.BadRequestError, match=expected_error):
        BedrockConverseMessagesProcessor.process_file_message(message)


def _get_gemini_function_response_inline_data_parts(result):
    assert isinstance(result, list), "expected Gemini parts list"
    assert len(result) == 1, "multimodal function responses should stay in one part"
    function_response_part = result[0]
    assert "inline_data" not in function_response_part, "inline_data should be nested under function_response.parts"
    function_response = function_response_part["function_response"]
    nested_parts = function_response["parts"]
    return [part["inline_data"] for part in nested_parts if "inline_data" in part]


def test_ollama_pt_simple_messages():
    """Test basic functionality with simple text messages"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "assistant", "content": "How can I help you?"},
        {"role": "user", "content": "Hello"},
    ]

    result = ollama_pt(model="llama2", messages=messages)

    expected_prompt = (
        "### System:\nYou are a helpful assistant\n\n### Assistant:\nHow can I help you?\n\n### User:\nHello\n\n"
    )
    assert isinstance(result, dict)
    assert result["prompt"] == expected_prompt
    assert result["images"] == []


@pytest.mark.parametrize(
    ("assistant_content", "rendered_prefix"),
    [("", ""), ("Checking calc.py", "Checking calc.py\n")],
)
def test_ollama_pt_renders_tool_calls_in_function_prompt_format(assistant_content: str, rendered_prefix: str):
    messages: Final = [
        {"role": "user", "content": "Fix calc.py"},
        {
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"filePath": "calc.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "def add(a, b): return a - b"},
    ]

    result: Final = ollama_pt(model="gemma4:31b", messages=messages)

    assert result["prompt"] == (
        "### User:\nFix calc.py\n\n"
        f'### Assistant:\n{rendered_prefix}{{"name": "read", "arguments": {{"filePath": "calc.py"}}}}\n\n'
        "### User:\ndef add(a, b): return a - b\n\n"
    )


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


def _ollama_tool_turn(*results: object) -> list[dict]:
    call: Final = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
    return [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        *({"role": "tool", "tool_call_id": "call_1", "content": result} for result in results),
    ]


@pytest.mark.parametrize(
    ("results", "forwarded"),
    [
        pytest.param(("Paris: 22 degrees", "Sky: clear"), "Paris: 22 degrees\nSky: clear", id="two-tool-messages"),
        pytest.param(
            ([{"type": "text", "text": "Paris: 22 degrees"}, {"type": "text", "text": "clear skies"}],),
            "Paris: 22 degrees\nclear skies",
            id="text-parts-of-one-tool-message",
        ),
        pytest.param(
            ([{"type": "text", "text": "Paris: 22 degrees"}], "Sky: clear"),
            "Paris: 22 degrees\nSky: clear",
            id="text-part-then-string",
        ),
        pytest.param(
            ([{"type": "text", "text": ""}, {"type": "text", "text": "clear skies"}],),
            "clear skies",
            id="empty-text-part-adds-no-blank-line",
        ),
    ],
)
def test_ollama_pt_separates_merged_tool_results_with_a_newline(results: tuple[object, ...], forwarded: str):
    result: Final = ollama_pt(model="llama2", messages=_ollama_tool_turn(*results))

    assert isinstance(result, dict)
    assert result["prompt"].endswith(f"### User:\n{forwarded}\n\n"), result["prompt"]


@pytest.mark.parametrize("content", [22, 22.5, True, {"temperature": 22}], ids=type)
def test_ollama_pt_rejects_non_text_tool_content_as_a_bad_request(content: object):
    with pytest.raises(litellm.BadRequestError) as excinfo:
        ollama_pt(model="llama2", messages=_ollama_tool_turn(content))

    assert excinfo.value.status_code == 400
    assert "content" in excinfo.value.message
    assert "tool message at index 2" in excinfo.value.message
    assert type(content).__name__ in excinfo.value.message


@pytest.mark.parametrize(
    ("part", "expected_detail"),
    (
        ({"type": "image_url", "image_url": None}, "NoneType image_url"),
        ({"type": "text", "text": 22}, "int text part"),
        ({"type": "text"}, "text part with no text"),
        ({"type": "image_url"}, "image_url part with no image_url"),
        ({"type": "image_url", "image_url": {"detail": "high"}}, "image_url object without a url string"),
        ("hello", "str content part"),
    ),
    ids=(
        "none-image-url",
        "int-text",
        "text-without-text",
        "image-url-without-image-url",
        "image-url-object-without-url",
        "str-part",
    ),
)
def test_ollama_pt_rejects_a_malformed_content_part_as_a_bad_request(part: object, expected_detail: str):
    messages: Final = [{"role": "user", "content": [part]}]

    with pytest.raises(litellm.BadRequestError) as excinfo:
        ollama_pt(model="llava", messages=messages)

    assert excinfo.value.status_code == 400
    assert "user message at index 0" in excinfo.value.message
    assert expected_detail in excinfo.value.message


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
    assert result[1]["content"][0]["reasoningContent"]["reasoningText"]["text"] == "This is a test thinking block"


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
            assert block["text"].strip(), f"Bedrock Converse rejects blank-text ContentBlocks; got {block!r}"
    # toolUse blocks must still be present
    tool_use_blocks = [b for b in assistant_blocks[0]["content"] if "toolUse" in b]
    assert len(tool_use_blocks) == 2


@pytest.mark.parametrize(
    "thinking_block",
    [
        {"type": "thinking", "thinking": "oss reasoning", "signature": None},
        {"type": "thinking", "thinking": "oss reasoning", "signature": ""},
        {"type": "thinking", "thinking": "oss reasoning"},
        {"type": "thinking", "thinking": "openai reasoning", "signature": "litellm_encrypted_reasoning:gAAAA"},
        {"type": "redacted_thinking", "data": "litellm_encrypted_reasoning:gAAAA"},
    ],
    ids=[
        "null_signature",
        "empty_signature",
        "missing_signature",
        "encrypted_reasoning_signature",
        "encrypted_reasoning_redacted_data",
    ],
)
def test_anthropic_messages_pt_drops_unsignable_thinking_block(thinking_block):
    """Open-source reasoning models (DeepSeek-R1, Qwen, etc.) emit thinking blocks
    with no Anthropic signature. Anthropic verifies the signature cryptographically,
    so replaying a null/empty/missing-signature thinking block is rejected with
    400 ... thinking.signature.str: Input should be a valid string.
    anthropic_messages_pt must drop the unsignable thinking block while preserving
    the assistant's answer text. Regression for LIT-4007.
    """
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "2+2 equals 4.",
            "thinking_blocks": [thinking_block],
        },
        {"role": "user", "content": "Now what is 3+3?"},
    ]

    result = anthropic_messages_pt(messages=messages, model="claude-sonnet-4-6", llm_provider="anthropic")

    assistant = next(m for m in result if m["role"] == "assistant")
    content = assistant["content"]
    assert all(block.get("type") not in ("thinking", "redacted_thinking") for block in content), (
        f"unsignable thinking block must be dropped, got {content!r}"
    )
    assert any(block.get("type") == "text" and block.get("text") == "2+2 equals 4." for block in content), (
        f"assistant answer text must be preserved, got {content!r}"
    )


def test_anthropic_messages_pt_keeps_signed_thinking_block():
    """A genuine Anthropic round-trip still holds its original signature, so that
    thinking block must be forwarded unchanged (we only drop unsignable blocks).
    Regression for LIT-4007.
    """
    signed_block = {
        "type": "thinking",
        "thinking": "genuine anthropic reasoning",
        "signature": "ErcBCkgIValidSignatureBytes",
    }
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "2+2 equals 4.",
            "thinking_blocks": [signed_block],
        },
        {"role": "user", "content": "Now what is 3+3?"},
    ]

    result = anthropic_messages_pt(messages=messages, model="claude-sonnet-4-6", llm_provider="anthropic")

    assistant = next(m for m in result if m["role"] == "assistant")
    thinking_blocks = [b for b in assistant["content"] if b.get("type") == "thinking"]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0]["signature"] == "ErcBCkgIValidSignatureBytes"
    assert thinking_blocks[0]["thinking"] == "genuine anthropic reasoning"


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


def test_convert_to_azure_openai_messages_strips_litellm_format_from_file_and_image():
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_azure_openai_messages,
    )
    from litellm.types.llms.openai import AllMessageValues

    input: list[AllMessageValues] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {"file_id": "assistant-xyz", "format": "application/pdf"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://x/y.png", "format": "image/png"},
                },
            ],
        }
    ]

    output = convert_to_azure_openai_messages(input)

    content = output[0].get("content")
    assert content[0]["file"] == {"file_id": "assistant-xyz"}
    assert content[1]["image_url"] == {"url": "https://x/y.png"}


def test_bedrock_validate_format_image_or_video():
    """Test the validate_format method for images, videos, and documents"""

    # Test valid image formats
    valid_image_formats = ["png", "jpeg", "gif", "webp"]
    for format in valid_image_formats:
        result = BedrockImageProcessor.validate_format(f"image/{format}", format)
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
        result = BedrockImageProcessor.validate_format(f"video/{format}", format)
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
        result = BedrockImageProcessor.validate_format(mime, mime.split("/")[1])
        assert result == expected, f"Expected {expected}, got {result}"


def test_bedrock_get_document_format_fallback_mimes():
    """
    Test the _get_document_format method with fallback MIME types for DOCX and XLSX.

    This tests the fallback mechanism when mimetypes.guess_all_extensions returns empty results,
    which can happen in Docker containers where mimetypes depends on OS-installed MIME types.
    """

    # Test DOCX fallback
    docx_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
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
    docx_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    supported_formats = ["pdf", "docx", "xlsx", "csv"]

    # Test normal mimetypes behavior (should not hit fallback)
    result = BedrockImageProcessor._get_document_format(mime_type=docx_mime, supported_doc_formats=supported_formats)
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

    content_block = await BedrockImageProcessor.process_image_async(image_url=image_url, format=None)
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
    assert isinstance(items_schema, dict), "Items schema should be a dict after unpacking"
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
    inline_parts = _get_gemini_function_response_inline_data_parts(result)
    assert len(inline_parts) == 1

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
    inline_parts = _get_gemini_function_response_inline_data_parts(result2)
    assert len(inline_parts) == 1


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
    inline_parts = _get_gemini_function_response_inline_data_parts(result)
    assert len(inline_parts) == 1, "expected exactly one inline_data part"
    assert inline_parts[0]["mime_type"] == "image/png"
    assert inline_parts[0]["data"] == tiny_png_b64


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
    inline_parts = _get_gemini_function_response_inline_data_parts(result)
    assert len(inline_parts) == 2, f"expected 2 inline_data parts, got {len(inline_parts)}"
    mime_types = {p["mime_type"] for p in inline_parts}
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
    inline_parts = _get_gemini_function_response_inline_data_parts(result)
    assert len(inline_parts) == 1, "data-URL image string was not converted to inline_data"
    assert inline_parts[0]["mime_type"] == "image/png"
    assert inline_parts[0]["data"] == tiny_png_b64


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
    inline_parts = _get_gemini_function_response_inline_data_parts(result)
    assert len(inline_parts) == 1
    assert inline_parts[0]["mime_type"] == "image/png", (
        f"expected clean 'image/png', got '{inline_parts[0]['mime_type']}'"
    )


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


def test_bedrock_tools_pt_strict_parameter():
    """Regression for strict tools on the Bedrock Converse path.

    Claude on Bedrock honours strict in toolSpec (with additionalProperties, which
    Bedrock requires alongside strict); without forwarding it the model ignores the
    enum constraint the caller asked for. Every other Bedrock family (Nova, Llama,
    GPT-OSS) rejects the strict field, so it must only be forwarded for Claude.
    """
    tools_with_strict = [
        {
            "type": "function",
            "function": {
                "name": "generate_sql",
                "strict": True,
                "description": "Generate a SQL query",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    result = _bedrock_tools_pt(tools_with_strict, model="anthropic.claude-sonnet-4-5-20250929-v1:0")
    assert result[0]["toolSpec"]["strict"] is True
    assert result[0]["toolSpec"]["inputSchema"]["json"]["additionalProperties"] is False

    result = _bedrock_tools_pt(tools_with_strict, model="us.amazon.nova-micro-v1:0")
    assert "strict" not in result[0]["toolSpec"]
    assert "additionalProperties" not in result[0]["toolSpec"]["inputSchema"]["json"]

    tools_without_strict = [
        {
            "type": "function",
            "function": {
                "name": "generate_sql",
                "description": "Generate a SQL query",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    result = _bedrock_tools_pt(tools_without_strict, model="anthropic.claude-sonnet-4-5-20250929-v1:0")
    assert "strict" not in result[0]["toolSpec"]
    assert "additionalProperties" not in result[0]["toolSpec"]["inputSchema"]["json"]


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
    base64_bytes, content_type = BedrockImageProcessor._post_call_image_processing(mock_response, image_url)

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
    base64_bytes, content_type = BedrockImageProcessor._post_call_image_processing(mock_response, image_url)

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
    base64_bytes, content_type = BedrockImageProcessor._post_call_image_processing(mock_response, image_url)

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
    base64_bytes, content_type = BedrockImageProcessor._post_call_image_processing(mock_response, image_url)

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
    base64_bytes, content_type = BedrockImageProcessor._post_call_image_processing(mock_response, image_url)

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

    with pytest.raises(ValueError, match="Unable to determine content type from URL: https") as excinfo:
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
    _, content_type_jpg = BedrockImageProcessor._post_call_image_processing(mock_response, image_url_jpg)
    assert content_type_jpg == "image/jpeg"

    # Test with .jpeg extension
    image_url_jpeg = "https://example.com/photo.jpeg"
    _, content_type_jpeg = BedrockImageProcessor._post_call_image_processing(mock_response, image_url_jpeg)
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
    base64_bytes, content_type = BedrockImageProcessor._post_call_image_processing(mock_response, pdf_url)

    assert content_type == "application/pdf"
    assert base64_bytes == base64.b64encode(pdf_content).decode("utf-8")


def test_bedrock_image_processor_content_type_document_formats():
    """
    Test that _post_call_image_processing handles various document formats
    """

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
        _, content_type = BedrockImageProcessor._post_call_image_processing(mock_response, url)
        assert content_type == expected_mime, f"Expected {expected_mime} for {url}, got {content_type}"


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

    base64_bytes, content_type = BedrockImageProcessor._post_call_image_processing(mock_response, s3_url)

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
    base64_with_newlines = "\n".join([base64_content[i : i + 64] for i in range(0, len(base64_content), 64)])
    base64_with_spaces = " ".join([base64_content[i : i + 32] for i in range(0, len(base64_content), 32)])

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
    assert re.match(pattern, document_name), f"Document name format mismatch: {document_name}"


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
    result2 = config._map_web_search_options({"search_context_size": "high"}, "us.amazon.nova-premier-v1:0")

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


def test_bedrock_tools_pt_drops_unmappable_responses_builtin_tools():
    """
    Regression for LIT-3858: Responses built-in tools (image_generation, namespace,
    tool_search, custom) have no Bedrock toolSpec equivalent. They must be dropped, not
    emitted as junk ``litellm_unnamed_tool_N`` toolSpecs the model can hallucinate calls to.
    Mappable ``function`` and Anthropic ``input_schema`` tools must survive untouched.
    """
    from litellm.litellm_core_utils.prompt_templates.factory import _bedrock_tools_pt

    tools = [
        {
            "type": "function",
            "function": {
                "name": "noop",
                "description": "x",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {"type": "image_generation", "output_format": "png"},
        {"type": "namespace", "name": "grp", "description": "g", "tools": []},
        {"type": "custom", "name": "free_form"},
    ]

    result = _bedrock_tools_pt(tools=tools, model="anthropic.claude-sonnet-4-5-20250929-v1:0")

    names = [block["toolSpec"]["name"] for block in result if "toolSpec" in block]
    assert names == ["noop"]
    assert not any(name.startswith("litellm_unnamed_tool_") for name in names)


def test_bedrock_tools_pt_keeps_anthropic_input_schema_tools():
    """
    The drop guard for unmappable tools must not regress Anthropic Messages format tools,
    which carry an ``input_schema`` instead of an OpenAI ``function`` key.
    """
    from litellm.litellm_core_utils.prompt_templates.factory import _bedrock_tools_pt

    tools = [
        {
            "type": "image_generation",
            "output_format": "png",
        },
        {
            "name": "lookup",
            "description": "look something up",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        },
    ]

    result = _bedrock_tools_pt(tools=tools, model="anthropic.claude-sonnet-4-5-20250929-v1:0")

    names = [block["toolSpec"]["name"] for block in result if "toolSpec" in block]
    assert names == ["lookup"]


def test_convert_to_anthropic_tool_result_image_with_cache_control():
    """
    A cache_control on an image inside a tool message lands on the tool_result block,
    since Anthropic rejects cache_control within tool_result.content.
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

    assert result["content"][1]["type"] == "image"
    assert result["content"][1]["source"]["type"] == "base64"
    assert result["content"][1]["source"]["media_type"] == "image/jpeg"
    assert result["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in block for block in result["content"])


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
    assert [block["type"] for block in result["content"]] == ["text", "image", "text", "image"]
    assert result["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in block for block in result["content"])


def test_convert_to_anthropic_tool_result_last_block_marker_wins():
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_anthropic_tool_result,
    )

    message = {
        "role": "tool",
        "tool_call_id": "call_ttl",
        "content": [
            {"type": "text", "text": "older", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            {"type": "text", "text": "newest", "cache_control": {"type": "ephemeral"}},
        ],
    }

    result = convert_to_anthropic_tool_result(message)

    assert result["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in block for block in result["content"])


def test_convert_to_anthropic_tool_result_message_marker_wins_over_block_marker():
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_anthropic_tool_result,
    )

    message = {
        "role": "tool",
        "tool_call_id": "call_msg",
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
        "content": [{"type": "text", "text": "out", "cache_control": {"type": "ephemeral"}}],
    }

    result = convert_to_anthropic_tool_result(message)

    assert result["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "cache_control" not in result["content"][0]


def test_anthropic_messages_pt_tool_text_block_marker_stays_out_of_tool_result_content():
    from litellm.litellm_core_utils.prompt_templates.factory import anthropic_messages_pt

    messages = [
        {"role": "user", "content": "list files"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "toolu_01", "type": "function", "function": {"name": "ls", "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_01",
            "content": [
                {"type": "text", "text": "file1"},
                {"type": "text", "text": "file2", "cache_control": {"type": "ephemeral"}},
            ],
        },
    ]

    anthropic_messages = anthropic_messages_pt(model="claude-sonnet-4-5", messages=messages, llm_provider="anthropic")

    tool_result = anthropic_messages[-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in block for block in tool_result["content"])


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

    assert result["content"][0]["type"] == "image"
    assert result["content"][0]["source"]["type"] == "url"
    assert result["content"][0]["source"]["url"] == "https://example.com/image.jpg"
    assert result["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in result["content"][0]


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
                        "tool_references": [{"type": "tool_reference", "tool_name": "get_time"}],
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
    server_tool_use_block = next(b for b in assistant_msg["content"] if b.get("type") == "server_tool_use")
    assert server_tool_use_block["id"] == "srvtoolu_01ABC123"
    assert server_tool_use_block["name"] == "tool_search_tool_regex"
    assert server_tool_use_block["input"] == {"query": ".*time.*"}

    # Verify tool_search_tool_result block is preserved
    assert "tool_search_tool_result" in content_types
    tool_result_block = next(b for b in assistant_msg["content"] if b.get("type") == "tool_search_tool_result")
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
                    {"$ref": "#/$defs/Expression"},  # Circular: Operand -> Expression -> Operand
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


def test_anthropic_messages_pt_file_block_cache_control_with_explicit_provider():
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
    assert "cache_control" in file_block, "cache_control should be preserved on file/document content blocks"
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


_BEDROCK_TOOL_USE_ID_RE = re.compile(r"^[a-zA-Z0-9_.:-]{1,64}$")


@pytest.mark.parametrize(
    "tool_call_id",
    [
        "call_" + "x" * 100,
        "call|with|pipes",
        "call_" + "y" * 60 + "|end",
        "call:ok.dots-and_under",
        "",
    ],
)
def test_bedrock_tool_use_id_is_sanitized_consistently_for_invoke_and_result(tool_call_id):
    """
    Regression test for https://github.com/BerriAI/litellm/issues/34239: client-minted
    tool_call ids longer than 64 chars or with chars outside [a-zA-Z0-9_.:-] made Bedrock
    return a 400. The invoke and result paths must produce the same valid toolUseId so the
    toolUse/toolResult pair still correlates.
    """
    invoke = _convert_to_bedrock_tool_call_invoke(
        [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "Boston"}'},
            }
        ]
    )
    result = _convert_to_bedrock_tool_call_result(
        {"tool_call_id": tool_call_id, "role": "tool", "name": "get_weather", "content": "sunny"}
    )
    tool_use_id = invoke[0]["toolUse"]["toolUseId"]
    assert _BEDROCK_TOOL_USE_ID_RE.match(tool_use_id)
    assert result["toolResult"]["toolUseId"] == tool_use_id


def test_bedrock_tool_use_id_valid_ids_pass_through_unchanged():
    result = _convert_to_bedrock_tool_call_result(
        {"tool_call_id": "tooluse_Ab.c:1-2_3", "role": "tool", "name": "f", "content": "ok"}
    )
    assert result["toolResult"]["toolUseId"] == "tooluse_Ab.c:1-2_3"


def test_bedrock_tool_use_id_truncation_keeps_distinct_ids_distinct():
    prefix = "call_" + "z" * 70
    ids = {
        _convert_to_bedrock_tool_call_result(
            {"tool_call_id": f"{prefix}{suffix}", "role": "tool", "name": "f", "content": "ok"}
        )["toolResult"]["toolUseId"]
        for suffix in ("a", "b")
    }
    assert len(ids) == 2
    assert all(len(i) == 64 for i in ids)


def test_bedrock_tool_use_id_replaced_chars_do_not_collide_with_existing_ids():
    ids = {
        _convert_to_bedrock_tool_call_result({"tool_call_id": i, "role": "tool", "name": "f", "content": "ok"})[
            "toolResult"
        ]["toolUseId"]
        for i in ("call|x", "call_x")
    }
    assert len(ids) == 2


def test_bedrock_tool_call_invoke_concatenated_json_long_id_stays_within_limit():
    long_id = "call_" + "q" * 62
    result = _convert_to_bedrock_tool_call_invoke(
        [
            {
                "id": long_id,
                "type": "function",
                "function": {"name": "run", "arguments": '{"cmd":"a"}{"cmd":"b"}'},
            }
        ]
    )
    ids = [block["toolUse"]["toolUseId"] for block in result]
    assert len(ids) == 2
    assert len(set(ids)) == 2
    assert all(_BEDROCK_TOOL_USE_ID_RE.match(i) for i in ids)


@pytest.mark.parametrize(
    ("tool_call_id", "expected"),
    [
        ("call|with|pipes", "call_with_pipes"),
        ("call:ok.dots", "call_ok_dots"),
        ("call_" + "x" * 100, "call_" + "x" * 100),
        ("toolu_01AbC-xyz", "toolu_01AbC-xyz"),
        ("", "tool_use_id"),
    ],
)
def test_anthropic_tool_use_id_keeps_pattern_only_rewrite_with_no_cap_or_hash(tool_call_id, expected):
    result = convert_to_anthropic_tool_result({"role": "tool", "tool_call_id": tool_call_id, "content": "ok"})
    assert result["tool_use_id"] == expected


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
    assert result[0]["toolUse"]["input"] == {"command": ["curl", "-i", "http://localhost:9009", "-m", "10"]}

    # Subsequent blocks get suffixed ids
    assert result[1]["toolUse"]["toolUseId"] == "tooluse_L7I3TewYAUhoheJZQEuwVN_1"
    assert result[1]["toolUse"]["name"] == "shell"
    assert result[1]["toolUse"]["input"] == {"command": ["curl", "-i", "http://localhost:9009/robots.txt", "-m", "5"]}

    assert result[2]["toolUse"]["toolUseId"] == "tooluse_L7I3TewYAUhoheJZQEuwVN_2"
    assert result[2]["toolUse"]["name"] == "shell"
    assert result[2]["toolUse"]["input"] == {"command": ["curl", "-i", "http://localhost:9009/sitemap.xml", "-m", "5"]}


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


def test_bedrock_tool_call_invoke_malformed_json_does_not_raise():
    """
    Regression for https://github.com/BerriAI/litellm/issues/18667.

    When the model emits malformed JSON in tool-call arguments (here a
    missing comma between keys), replaying that history must NOT raise
    `Unable to convert openai tool calls ... Expecting ',' delimiter`.
    It degrades to an empty-object input so the conversation can continue.
    """
    tool_calls = [
        {
            "id": "toolu_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Boston" "unit": "celsius"}',
            },
        }
    ]
    result = _convert_to_bedrock_tool_call_invoke(tool_calls)
    assert len(result) == 1
    assert result[0]["toolUse"]["toolUseId"] == "toolu_abc123"
    assert result[0]["toolUse"]["name"] == "get_weather"
    assert result[0]["toolUse"]["input"] == {}


def test_bedrock_tool_call_invoke_salvages_valid_prefix_before_truncated_tail():
    """
    A valid leading object followed by a truncated tail keeps the valid
    object rather than dropping everything or raising.
    """
    tool_calls = [
        {
            "id": "call_partial",
            "type": "function",
            "function": {"name": "shell", "arguments": '{"cmd": "ls"}{"cmd":'},
        }
    ]
    result = _convert_to_bedrock_tool_call_invoke(tool_calls)
    assert len(result) == 1
    assert result[0]["toolUse"]["input"] == {"cmd": "ls"}


def test_bedrock_tool_call_invoke_mixed_turn_survives_one_malformed_call():
    """
    Regression for LIT-4574: an assistant turn with several tool calls where only one
    has malformed/truncated arguments must keep the valid calls intact and degrade just
    the bad one to empty input, instead of killing the entire turn.
    """
    tool_calls = [
        {
            "id": "t_good",
            "type": "function",
            "function": {
                "name": "good_tool",
                "arguments": '{"item_type": "email", "item_id": "AAMkAD=="}',
            },
        },
        {
            "id": "t_bad",
            "type": "function",
            "function": {"name": "bad_tool", "arguments": '{"item_type": "email"'},
        },
    ]
    result = _convert_to_bedrock_tool_call_invoke(tool_calls)
    tool_uses = [block["toolUse"] for block in result if "toolUse" in block]
    assert len(tool_uses) == 2
    by_name = {tool_use["name"]: tool_use for tool_use in tool_uses}
    assert by_name["good_tool"]["input"] == {"item_type": "email", "item_id": "AAMkAD=="}
    assert by_name["bad_tool"]["input"] == {}


def test_bedrock_tool_call_invoke_truncated_json_arguments():
    """
    Truncated tool call arguments (issue #35303) must not raise. A client replaying a
    partially streamed tool call would otherwise trigger a pre-network exception that the
    router maps to a retryable APIConnectionError and retries through the fallback graph.
    """
    tool_calls = [
        {
            "id": "tooluse_MAh2QLVjBRkvi5QJkLQ08V",
            "type": "function",
            "function": {
                "name": "replace_note_content",
                "arguments": '{"note_id": "999af35c-4061-4ece-8581-7d43fc988ba4", "title": "WG"',
            },
        }
    ]
    result = _convert_to_bedrock_tool_call_invoke(tool_calls)
    assert len(result) == 1
    assert result[0]["toolUse"]["toolUseId"] == "tooluse_MAh2QLVjBRkvi5QJkLQ08V"
    assert result[0]["toolUse"]["input"] == {}


def test_bedrock_tool_call_invoke_unconvertible_raises_non_retryable_bad_request():
    """
    Conversion failures are client input errors, so they must surface as a non-retryable
    BadRequestError instead of a bare Exception that maps to APIConnectionError, and the
    message must not embed the tool call payload (issue #35303).
    """
    tool_calls = [{"id": "call_bad", "type": "function", "function": None}]

    with pytest.raises(litellm.BadRequestError) as exc_info:
        _convert_to_bedrock_tool_call_invoke(tool_calls)

    assert exc_info.value.status_code == 400
    assert "call_bad" in str(exc_info.value)
    assert "function" not in str(exc_info.value).split("Received error=")[0]


def test_make_valid_bedrock_tool_name_preserves_hyphens():
    assert make_valid_bedrock_tool_name("my-tool") == "my-tool"
    assert (
        make_valid_bedrock_tool_name("CreateCaseKnowledgeArticle_foTWsqR6yDt-OnSsvR5e6Q")
        == "CreateCaseKnowledgeArticle_foTWsqR6yDt-OnSsvR5e6Q"
    )


def test_bedrock_tool_name_sanitized_consistently_in_tools_and_tool_use():
    """toolSpec and toolUse names must match after sanitization (issue #5007)."""
    raw_name = "foo@bar"
    tools = [
        {
            "type": "function",
            "function": {
                "name": raw_name,
                "description": "test",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    tool_spec_name = _bedrock_tools_pt(tools)[0]["toolSpec"]["name"]

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": raw_name, "arguments": "{}"},
        }
    ]
    tool_use_name = _convert_to_bedrock_tool_call_invoke(tool_calls)[0]["toolUse"]["name"]

    assert tool_spec_name == "foo_bar"
    assert tool_use_name == tool_spec_name


def test_bedrock_converse_messages_pt_tool_use_matches_tool_spec_hyphen_name():
    """Hyphenated tool names are preserved and consistent in multi-turn history."""
    tool_name = "my-tool"
    messages = [
        {"role": "user", "content": "call the tool"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_hyphen",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
    ]
    translated = _bedrock_converse_messages_pt(messages=messages, model="", llm_provider="")
    tool_use_blocks = [block for msg in translated for block in msg.get("content", []) if "toolUse" in block]
    assert len(tool_use_blocks) == 1
    assert tool_use_blocks[0]["toolUse"]["name"] == tool_name

    tool_spec_name = _bedrock_tools_pt(
        [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "test",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )[0]["toolSpec"]["name"]
    assert tool_spec_name == tool_name


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
        tool_results = [m for m in result if m.get("role") == "tool" and m.get("tool_call_id") == "call_abc123"]
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
        tool_results = [m for m in result if m.get("role") == "tool" and m.get("tool_call_id") == "call_X"]
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
        missing_results = [m for m in tool_results if m.get("tool_call_id") == "call_missing"]
        assert len(missing_results) == 1, (
            f"Expected 1 dummy result for call_missing (Case A), got {len(missing_results)}"
        )

        # Case D: call_duped should have exactly 1 result (the fresh one)
        duped_results = [m for m in tool_results if m.get("tool_call_id") == "call_duped"]
        assert len(duped_results) == 1, (
            f"Expected 1 result for call_duped after dedup (Case D), got {len(duped_results)}"
        )
        assert duped_results[0]["content"] == "fresh_result", (
            f"Expected last-wins 'fresh_result', got '{duped_results[0]['content']}'"
        )

        # Verify tool results immediately follow the assistant message
        asst_idx = next(i for i, m in enumerate(result) if m.get("role") == "assistant")
        tool_msgs_after_asst = [m for m in result[asst_idx + 1 :] if m.get("role") in ("tool", "function")]
        assert len(tool_msgs_after_asst) == 2, (
            f"Expected 2 tool results after assistant, got {len(tool_msgs_after_asst)}"
        )
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

    result = anthropic_messages_pt(messages, model="claude-sonnet-4-20250514", llm_provider="anthropic")

    content_blocks = result[0]["content"]
    assert len(content_blocks) == 2

    # Document block (from file) should preserve cache_control
    doc_block = content_blocks[0]
    assert doc_block["type"] == "document"
    assert "cache_control" in doc_block, "cache_control was dropped from file/document block"
    assert doc_block["cache_control"]["type"] == "ephemeral"

    # Text block should also preserve cache_control
    text_block = content_blocks[1]
    assert text_block["type"] == "text"
    assert "cache_control" in text_block
    assert text_block["cache_control"]["type"] == "ephemeral"


def test_add_cache_point_tool_block_passes_ttl_for_claude_4_5(monkeypatch):
    """
    Tools with cache_control ttl should preserve the ttl in the cachePoint
    block for Claude 4.5+ models on Bedrock, matching the behavior of system
    block cache_control.

    Without this fix, tool cachePoint is always {"type": "default"} (5m),
    while system blocks can have ttl="1h", violating Bedrock's non-increasing
    TTL ordering constraint (tools -> system -> messages).

    Ref: https://github.com/BerriAI/litellm/issues/XXXXX

    Forces the bundled local cost map so ttl eligibility (driven by
    `cache_creation_input_token_cost_above_1hr` in litellm.model_cost) reads
    this branch's pricing data rather than the network-fetched `main` copy,
    which lacks the fix until merge.
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        add_cache_point_tool_block,
    )

    old_env = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
    old_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    try:
        tool_with_1h = {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }

        # Claude 4.5 model: ttl should be preserved
        result = add_cache_point_tool_block(tool_with_1h, model="jp.anthropic.claude-opus-4-7")
        assert result is not None
        assert result["cachePoint"]["type"] == "default"
        assert result["cachePoint"]["ttl"] == "1h"

        # Claude 4.5 model with 5m ttl: also preserved
        tool_with_5m = {
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        }
        result_5m = add_cache_point_tool_block(tool_with_5m, model="jp.anthropic.claude-opus-4-7")
        assert result_5m is not None
        assert result_5m["cachePoint"]["ttl"] == "5m"

        # Older model: ttl should be stripped
        result_old = add_cache_point_tool_block(tool_with_1h, model="anthropic.claude-3-5-sonnet-20241022-v2:0")
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
        result_no_ttl = add_cache_point_tool_block(tool_no_ttl, model="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        assert result_no_ttl is not None
        assert result_no_ttl["cachePoint"]["type"] == "default"
        assert "ttl" not in result_no_ttl["cachePoint"]
    finally:
        litellm.model_cost = old_cost
        if old_env is None:
            os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
        else:
            monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", old_env)


def test_add_cache_point_tool_block_stands_down_for_model_without_prompt_caching(monkeypatch):
    """A tool carrying cache_control must not become a cachePoint for a Bedrock model
    whose cost-map entry lacks prompt caching support, since Bedrock rejects the whole
    request. An unmapped id keeps emitting so ARN deployments do not lose caching."""
    from litellm.litellm_core_utils.prompt_templates.factory import (
        add_cache_point_tool_block,
    )

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    tool = {"cache_control": {"type": "ephemeral"}}

    assert add_cache_point_tool_block(tool, model="nvidia.nemotron-super-3-120b") is None
    assert add_cache_point_tool_block(tool, model="us.nvidia.nemotron-super-3-120b") is None
    assert add_cache_point_tool_block(
        tool, model="arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"
    ) == {"cachePoint": {"type": "default"}}
    assert add_cache_point_tool_block(tool, model="us.anthropic.claude-sonnet-4-5-20250929-v1:0") == {
        "cachePoint": {"type": "default"}
    }


def test_bedrock_tools_pt_passes_ttl_for_claude_4_5(monkeypatch):
    """
    End-to-end: _bedrock_tools_pt should produce cachePoint blocks with ttl
    for Claude 4.5+ models when tools have cache_control with ttl.

    Forces the bundled local cost map so ttl eligibility (driven by
    `cache_creation_input_token_cost_above_1hr` in litellm.model_cost) reads
    this branch's pricing data rather than the network-fetched `main` copy,
    which lacks the fix until merge.
    """
    from litellm.litellm_core_utils.prompt_templates.factory import _bedrock_tools_pt

    old_env = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
    old_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    try:
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
        result = _bedrock_tools_pt(tools, model="jp.anthropic.claude-opus-4-7")
        cache_blocks = [b for b in result if "cachePoint" in b]
        assert len(cache_blocks) == 1
        assert cache_blocks[0]["cachePoint"]["ttl"] == "1h"

        # Older model: cachePoint should not have ttl
        result_old = _bedrock_tools_pt(tools, model="anthropic.claude-3-5-sonnet-20241022-v2:0")
        cache_blocks_old = [b for b in result_old if "cachePoint" in b]
        assert len(cache_blocks_old) == 1
        assert "ttl" not in cache_blocks_old[0]["cachePoint"]
    finally:
        litellm.model_cost = old_cost
        if old_env is None:
            os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
        else:
            monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", old_env)


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


def test_convert_to_bedrock_tool_call_result_maps_tool_references_to_text() -> None:
    tool_reference_message: Final[ChatCompletionToolMessage] = {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": [
            {"type": "tool_reference", "tool_name": "WebFetch"},
            {"type": "tool_reference", "tool_name": "WebSearch"},
        ],
    }
    mixed_message: Final[ChatCompletionToolMessage] = {
        "role": "tool",
        "tool_call_id": "toolu_2",
        "content": [
            {"type": "text", "text": "Loaded tools:"},
            {"type": "tool_reference", "tool_name": "WebSearch"},
        ],
    }

    tool_reference_result: Final = _convert_to_bedrock_tool_call_result(tool_reference_message)
    mixed_result: Final = _convert_to_bedrock_tool_call_result(mixed_message)

    assert tool_reference_result == {
        "toolResult": {
            "toolUseId": "toolu_1",
            "content": [{"text": "WebFetch"}, {"text": "WebSearch"}],
        }
    }
    assert mixed_result == {
        "toolResult": {
            "toolUseId": "toolu_2",
            "content": [{"text": "Loaded tools:"}, {"text": "WebSearch"}],
        }
    }


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

        result = _bedrock_converse_messages_pt(messages, "anthropic.claude-sonnet-4-6", "bedrock")

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

    result1 = _bedrock_converse_messages_pt(messages, "anthropic.claude-sonnet-4-6", "bedrock")
    result2 = _bedrock_converse_messages_pt(messages, "anthropic.claude-sonnet-4-6", "bedrock")

    name1 = result1[0]["content"][0]["document"]["name"]
    name2 = result2[0]["content"][0]["document"]["name"]
    assert name1 == name2


def test_bedrock_converse_messages_pt_renames_duplicate_document_names():
    """
    The same document in multiple turns must not produce duplicate names;
    Bedrock rejects requests with "Messages can not contain duplicate
    document names". The first occurrence keeps its hash-based name and
    later occurrences get a deterministic positional suffix.
    """
    document_block = {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": "dGVzdA==",
        },
    }
    messages = [
        {
            "role": "user",
            "content": [document_block, {"type": "text", "text": "summarize this"}],
        },
        {"role": "assistant", "content": "It says test."},
        {
            "role": "user",
            "content": [document_block, {"type": "text", "text": "summarize again"}],
        },
    ]

    result1 = _bedrock_converse_messages_pt(messages, "anthropic.claude-sonnet-4-6", "bedrock")
    result2 = _bedrock_converse_messages_pt(messages, "anthropic.claude-sonnet-4-6", "bedrock")

    names1 = [block["document"]["name"] for message in result1 for block in message["content"] if "document" in block]
    names2 = [block["document"]["name"] for message in result2 for block in message["content"] if "document" in block]

    assert len(names1) == 2
    assert len(set(names1)) == 2
    assert names1[1] == f"{names1[0]}_2"
    assert names1 == names2

    single_turn = _bedrock_converse_messages_pt([messages[0]], "anthropic.claude-sonnet-4-6", "bedrock")
    assert names1[0] == single_turn[0]["content"][0]["document"]["name"]


def test_rename_duplicate_bedrock_document_names_skips_organic_suffixes():
    """
    A renamed duplicate must not collide with a document whose organic name
    already carries the would-be suffix (e.g. an existing ``report_2``),
    regardless of whether that document appears before or after the rename.
    """

    def _contents(names):
        return [
            {
                "role": "user",
                "content": [{"document": {"name": name}} for name in names],
            }
        ]

    def _names(contents):
        return [block["document"]["name"] for block in contents[0]["content"]]

    organic_first = _rename_duplicate_bedrock_document_names(_contents(["report", "report_2", "report"]))
    assert _names(organic_first) == ["report", "report_2", "report_3"]

    organic_last = _rename_duplicate_bedrock_document_names(_contents(["report", "report", "report_2"]))
    assert _names(organic_last) == ["report", "report_3", "report_2"]


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
        _bedrock_converse_messages_pt(messages, "anthropic.claude-sonnet-4-6", "bedrock")


def _collect_cache_points(blocks):
    return [block["cachePoint"] for message in blocks for block in message["content"] if "cachePoint" in block]


@pytest.mark.parametrize(
    "messages",
    [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "conversation history",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
        ],
        [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "assistant reply",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
        ],
    ],
)
def test_bedrock_converse_message_level_cache_point_preserves_ttl(messages):
    """
    Regression for https://github.com/BerriAI/litellm/issues/32154: message-level
    cache_control ttl was silently dropped because the message-level
    _get_cache_point_block call sites never passed model=, so multi-turn prefixes
    fell back to the 5m default while the system prompt kept 1h, churning the
    cache every turn on models like Opus 4.8.
    """
    result = _bedrock_converse_messages_pt(
        messages=messages,
        model="eu.anthropic.claude-opus-4-8",
        llm_provider="bedrock",
    )

    cache_points = _collect_cache_points(result)
    assert cache_points == [{"type": "default", "ttl": "1h"}]


@pytest.mark.asyncio
async def test_bedrock_converse_message_level_cache_point_preserves_ttl_async():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "conversation history",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
        },
    ]

    result = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages=messages,
        model="eu.anthropic.claude-opus-4-8",
        llm_provider="bedrock",
    )

    assert _collect_cache_points(result) == [{"type": "default", "ttl": "1h"}]


def _n_choices_response(*names_per_choice):
    from types import SimpleNamespace

    choices = [
        SimpleNamespace(
            message=SimpleNamespace(
                tool_calls=[SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=name, arguments="{}"))]
            )
        )
        for i, name in enumerate(names_per_choice)
    ]
    return SimpleNamespace(choices=choices)


def test_get_tool_calls_from_response_defaults_to_primary_choice_only():
    from litellm.litellm_core_utils.prompt_templates.factory import get_tool_calls_from_response

    response = _n_choices_response("tool_alpha", "tool_beta")

    assert [tc["name"] for tc in get_tool_calls_from_response(response)] == ["tool_alpha"]


def test_get_tool_calls_from_response_include_all_choices_reads_every_choice():
    from litellm.litellm_core_utils.prompt_templates.factory import get_tool_calls_from_response

    response = _n_choices_response("tool_alpha", "tool_beta")

    names = [tc["name"] for tc in get_tool_calls_from_response(response, include_all_choices=True)]
    assert names == ["tool_alpha", "tool_beta"]


def test_get_tool_calls_from_response_silences_redacted_arguments(caplog):
    from litellm.litellm_core_utils.prompt_templates.factory import (
        get_tool_calls_from_response,
    )

    response: Final = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "Read",
                                "arguments": "redacted-by-litellm",
                            },
                        }
                    ]
                }
            }
        ]
    }

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        tool_calls: Final = get_tool_calls_from_response(response)

    assert tool_calls == [{"id": "call_1", "name": "Read", "arguments": {}}]
    assert "Failed to parse tool call arguments" not in caplog.text


def test_get_tool_calls_from_response_warns_for_malformed_arguments(caplog):
    from litellm.litellm_core_utils.prompt_templates.factory import (
        get_tool_calls_from_response,
    )

    response: Final = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "Read",
                                "arguments": "not-json",
                            },
                        }
                    ]
                }
            }
        ]
    }

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        tool_calls: Final = get_tool_calls_from_response(response)

    assert tool_calls == [{"id": "call_1", "name": "Read", "arguments": {}}]
    assert "Failed to parse tool call arguments" in caplog.text


def _concatenated_json(*payloads: dict[str, object]) -> str:
    return "".join(json.dumps(payload, separators=(",", ":")) for payload in payloads)


def _function_tool_call(call_id: str | None, name: str, arguments: str) -> dict[str, object]:
    return {"id": call_id, "function": {"name": name, "arguments": arguments}}


def _chat_tool_response(*tool_calls: dict[str, object]) -> dict[str, object]:
    return {"choices": [{"message": {"tool_calls": list(tool_calls)}}]}


def test_get_tool_calls_from_response_expands_distinct_concatenated_arguments(caplog):
    raw = '{"flag":true}{"box":"A","limit":50}'
    response: Final = _chat_tool_response(_function_tool_call("call_move", "move", raw))

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        tool_calls: Final = get_tool_calls_from_response(response)

    assert tool_calls == [
        {"id": "call_move", "name": "move", "arguments": {"flag": True}},
        {"id": "call_move__concat_1", "name": "move", "arguments": {"box": "A", "limit": 50}},
    ]
    assert "Recovered 2 tool call(s)" in caplog.text
    assert "move" in caplog.text
    assert "flag" not in caplog.text


def test_get_tool_calls_from_response_expands_responses_api_concatenated_arguments():
    response: Final = {
        "output": [
            {
                "type": "function_call",
                "call_id": "call_move",
                "name": "move",
                "arguments": '{"flag":true}{"box":"A","limit":50}',
            }
        ]
    }

    assert get_tool_calls_from_response(response) == [
        {"id": "call_move", "name": "move", "arguments": {"flag": True}},
        {"id": "call_move__concat_1", "name": "move", "arguments": {"box": "A", "limit": 50}},
    ]


def test_get_tool_calls_from_response_collapses_identical_concatenated_arguments():
    response: Final = _chat_tool_response(_function_tool_call("call_move", "move", '{"flag":true}' * 3))

    assert get_tool_calls_from_response(response) == [
        {"id": "call_move", "name": "move", "arguments": {"flag": True}},
    ]


def test_get_tool_calls_from_response_does_not_expand_a_valid_json_array():
    response: Final = _chat_tool_response(_function_tool_call("call_batch", "batch", '[{"a":1},{"b":2}]'))

    assert get_tool_calls_from_response(response) == [
        {"id": "call_batch", "name": "batch", "arguments": {}},
    ]


@pytest.mark.parametrize("arguments", ('{"a":1}{"b":', '0{"x":1}'))
def test_get_tool_calls_from_response_drops_partial_concatenated_arguments(arguments: str, caplog):
    response: Final = _chat_tool_response(_function_tool_call("call_move", "move", arguments))

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        tool_calls: Final = get_tool_calls_from_response(response)

    assert tool_calls == [{"id": "call_move", "name": "move", "arguments": {}}]
    assert "Failed to parse tool call arguments" in caplog.text


@pytest.mark.parametrize(("count", "expands"), ((8, True), (9, False)))
def test_get_tool_calls_from_response_caps_distinct_concatenated_arguments(count: int, expands: bool):
    raw = _concatenated_json(*({"n": index} for index in range(count)))
    response: Final = _chat_tool_response(_function_tool_call("call", "move", raw))

    tool_calls: Final = get_tool_calls_from_response(response)

    if expands:
        assert [call["id"] for call in tool_calls] == ["call", *(f"call__concat_{index}" for index in range(1, count))]
        assert [call["arguments"] for call in tool_calls] == [{"n": index} for index in range(count)]
        return
    assert tool_calls == [{"id": "call", "name": "move", "arguments": {}}]


def test_get_tool_calls_from_response_skips_concat_ids_taken_by_a_sibling():
    raw = _concatenated_json({"a": 1}, {"b": 2})
    response: Final = _chat_tool_response(
        _function_tool_call("call", "move", raw),
        _function_tool_call("call__concat_1", "look", '{"x":1}'),
    )

    assert [call["id"] for call in get_tool_calls_from_response(response)] == [
        "call",
        "call__concat_2",
        "call__concat_1",
    ]


def test_get_tool_calls_from_response_keeps_sanitized_concat_ids_distinct():
    raw = _concatenated_json({"a": 1}, {"b": 2})
    response: Final = _chat_tool_response(
        _function_tool_call("a:b", "move", raw),
        _function_tool_call("a_b__concat_1", "look", '{"x":1}'),
    )

    ids: Final = [call["id"] for call in get_tool_calls_from_response(response)]
    sanitized: Final = [_sanitize_anthropic_tool_use_id(call_id) for call_id in ids if isinstance(call_id, str)]

    assert len(sanitized) == len(set(sanitized))
    assert ids == ["a:b", "a:b__concat_2", "a_b__concat_1"]


def test_get_tool_calls_from_response_bumps_suffix_when_sibling_sanitizes_onto_it():
    raw = _concatenated_json({"a": 1}, {"b": 2})
    response: Final = _chat_tool_response(
        _function_tool_call("a_b", "move", raw),
        _function_tool_call("a:b__concat_1", "look", '{"x":1}'),
    )

    ids: Final = [call["id"] for call in get_tool_calls_from_response(response)]
    sanitized: Final = [_sanitize_anthropic_tool_use_id(call_id) for call_id in ids if isinstance(call_id, str)]

    assert len(ids) == len(sanitized)
    assert len(sanitized) == len(set(sanitized))
    assert ids == ["a_b", "a_b__concat_2", "a:b__concat_1"]


def test_get_tool_calls_from_response_continues_concat_suffixes_per_sanitized_base():
    raw = _concatenated_json({"a": 1}, {"b": 2})
    response: Final = _chat_tool_response(
        _function_tool_call("x", "move", raw),
        _function_tool_call("x", "move", raw),
    )

    assert [call["id"] for call in get_tool_calls_from_response(response)] == [
        "x",
        "x__concat_1",
        "x",
        "x__concat_2",
    ]


def test_get_tool_calls_from_response_skips_a_run_of_reserved_concat_ids():
    raw = _concatenated_json({"a": 1}, {"b": 2})
    siblings: Final = tuple(_function_tool_call(f"call__concat_{index}", "look", '{"x":1}') for index in range(1, 51))
    response: Final = _chat_tool_response(_function_tool_call("call", "move", raw), *siblings)

    ids: Final = [call["id"] for call in get_tool_calls_from_response(response)]

    assert ids[0] == "call"
    assert ids[1] == "call__concat_51"


def test_get_tool_calls_from_response_reserves_concat_ids_across_choices():
    raw = _concatenated_json({"a": 1}, {"b": 2})
    response: Final = {
        "choices": [
            {"message": {"tool_calls": [_function_tool_call("call", "move", raw)]}},
            {"message": {"tool_calls": [_function_tool_call("call__concat_1", "look", '{"x":1}')]}},
        ]
    }

    assert [call["id"] for call in get_tool_calls_from_response(response, include_all_choices=True)] == [
        "call",
        "call__concat_2",
        "call__concat_1",
    ]


def test_get_tool_calls_from_response_does_not_invent_ids_for_a_missing_call_id():
    raw = _concatenated_json({"a": 1}, {"b": 2})
    response: Final = _chat_tool_response(_function_tool_call(None, "move", raw))

    tool_calls: Final = get_tool_calls_from_response(response)

    assert len(tool_calls) == 2
    assert all(call["id"] is None for call in tool_calls)
    assert [call["arguments"] for call in tool_calls] == [{"a": 1}, {"b": 2}]


def test_group_tool_exchanges_pairs_assistant_with_its_tool_rows():
    from litellm.litellm_core_utils.prompt_templates.factory import group_tool_exchanges

    messages = [
        {"role": "user", "content": "first turn"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tu_1", "type": "function", "function": {"name": "Read", "arguments": "{}"}},
                {"id": "tu_2", "type": "function", "function": {"name": "Grep", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "tu_1", "content": "file body"},
        {"role": "tool", "tool_call_id": "tu_2", "content": "matches"},
        {"role": "user", "content": "live instruction"},
    ]

    assert group_tool_exchanges(messages) == ((0,), (1, 2, 3), (4,))


def test_group_tool_exchanges_uses_ownership_not_adjacency():
    """A tool row answering some other call must not be swept into the exchange
    it happens to sit next to."""
    from litellm.litellm_core_utils.prompt_templates.factory import group_tool_exchanges

    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tu_1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "unrelated", "content": "not an answer to tu_1"},
        {"role": "tool", "tool_call_id": "tu_1", "content": "file body"},
    ]

    assert group_tool_exchanges(messages) == ((0,), (1,), (2,))


def test_group_tool_exchanges_assistant_without_tool_calls_stands_alone():
    from litellm.litellm_core_utils.prompt_templates.factory import group_tool_exchanges

    messages = [
        {"role": "assistant", "content": "no tools here"},
        {"role": "user", "content": "next"},
    ]

    assert group_tool_exchanges(messages) == ((0,), (1,))
    assert group_tool_exchanges([]) == ()


def test_group_tool_exchanges_is_linear_in_message_count():
    """Grouping runs on every guardrail write-back, over a message array the
    caller controls, so it has to stay linear. Accumulating groups by rebuilding
    a tuple each iteration made this O(n^2): 20k standalone messages took 312ms
    and 100k would take minutes. Linear finishes in single-digit ms, so this
    ceiling has ~200x headroom while a quadratic rewrite blows straight past it.
    """
    import time

    from litellm.litellm_core_utils.prompt_templates.factory import group_tool_exchanges

    messages = [{"role": "user", "content": "x"} for _ in range(100_000)]

    started = time.perf_counter()
    groups = group_tool_exchanges(messages)
    elapsed = time.perf_counter() - started

    assert len(groups) == 100_000
    assert elapsed < 3.0, f"grouping 100k messages took {elapsed:.2f}s; suspect superlinear accumulation"


_PDF_DATA_URI = "data:application/pdf;base64," + base64.b64encode(b"%PDF-1.4 regression fixture").decode()
_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _text_blocks(message):
    return [block["text"] for block in message["content"] if "text" in block]


def test_bedrock_converse_pdf_only_user_message_gets_text_block():
    """
    Regression for LIT-4523: Claude Code sends a PDF as a user turn whose only
    content is the document (an image_url part with a pdf data URI after the
    /v1/messages -> completion bridge). Bedrock Converse rejects any user
    message carrying a document without a sibling text block, so the builder
    must inject a placeholder text block.
    """
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": _PDF_DATA_URI}}],
        }
    ]

    result = _bedrock_converse_messages_pt(messages, "anthropic.claude-haiku-4-5", "bedrock")

    assert len(result) == 1
    assert any("document" in block for block in result[0]["content"])
    assert _text_blocks(result[0]) == [BEDROCK_DOCUMENT_PLACEHOLDER_TEXT]


def test_bedrock_converse_document_with_text_gets_no_extra_text_block():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _PDF_DATA_URI}},
                {"type": "text", "text": "summarize this"},
            ],
        }
    ]

    result = _bedrock_converse_messages_pt(messages, "anthropic.claude-haiku-4-5", "bedrock")

    assert _text_blocks(result[0]) == ["summarize this"]


def test_bedrock_converse_image_only_user_message_gets_no_text_block():
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": _PNG_DATA_URI}}],
        }
    ]

    result = _bedrock_converse_messages_pt(messages, "anthropic.claude-haiku-4-5", "bedrock")

    assert any("image" in block for block in result[0]["content"])
    assert _text_blocks(result[0]) == []


def test_bedrock_converse_tool_round_trip_document_injects_text_before_cache_point():
    """
    Claude Code shape: after a Read tool round trip, the document-only user
    turn (with cache_control) merges into the toolResult message. The injected
    text block must land before the trailing cachePoint so the cache boundary
    stays the final block, and earlier turns must stay untouched.
    """
    messages = [
        {"role": "user", "content": "read the pdf"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tooluse_pdf1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "tooluse_pdf1", "content": "read ok"},
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
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]

    result = _bedrock_converse_messages_pt(messages, "anthropic.claude-haiku-4-5", "bedrock")

    assert _text_blocks(result[0]) == ["read the pdf"]
    document_message = result[-1]
    block_keys = [next(iter(block)) for block in document_message["content"]]
    assert block_keys == ["toolResult", "document", "text", "cachePoint"]
    assert _text_blocks(document_message) == [BEDROCK_DOCUMENT_PLACEHOLDER_TEXT]


@pytest.mark.asyncio
async def test_bedrock_converse_pdf_only_user_message_gets_text_block_async():
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": _PDF_DATA_URI}}],
        }
    ]

    result = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages=messages,
        model="anthropic.claude-haiku-4-5",
        llm_provider="bedrock",
    )

    assert len(result) == 1
    assert any("document" in block for block in result[0]["content"])
    assert _text_blocks(result[0]) == [BEDROCK_DOCUMENT_PLACEHOLDER_TEXT]


def test_convert_to_anthropic_tool_result_keeps_tool_reference_blocks():
    from litellm.litellm_core_utils.prompt_templates.factory import convert_to_anthropic_tool_result

    result = convert_to_anthropic_tool_result(
        {
            "role": "tool",
            "tool_call_id": "toolu_01",
            "content": [
                {"type": "text", "text": "loaded"},
                {"type": "tool_reference", "tool_name": "WebFetch"},
            ],
        }
    )

    assert result == {
        "type": "tool_result",
        "tool_use_id": "toolu_01",
        "content": [
            {"type": "text", "text": "loaded"},
            {"type": "tool_reference", "tool_name": "WebFetch"},
        ],
    }


def test_convert_gemini_tool_call_result_answers_tool_reference_only_result():
    """Every Gemini function call needs a function response, even when the tool result carries no text.
    Fixes: https://github.com/BerriAI/litellm/issues/37462
    """
    result = convert_to_gemini_tool_call_result(
        message=ChatCompletionToolMessage(
            role="tool",
            tool_call_id="toolu_01",
            content=[{"type": "tool_reference", "tool_name": "WebFetch"}],
        ),
        last_message_with_tool_calls={
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "toolu_01",
                    "type": "function",
                    "function": {"name": "ToolSearch", "arguments": '{"query": "select:WebFetch"}'},
                }
            ],
        },
    )

    assert result == {"function_response": {"name": "ToolSearch", "response": {"content": ""}}}


def test_convert_to_anthropic_tool_invoke_degrades_unpaired_server_tool_use():
    """A replayed srvtoolu_ call whose server tool result is not available
    (e.g. the Responses bridge replays items without provider_specific_fields)
    must become a plain client tool_use so the client's tool_result can pair
    with it. A dangling server_tool_use makes Anthropic 400 the request with
    "unexpected `tool_use_id` found in `tool_result` blocks"."""
    from litellm.litellm_core_utils.prompt_templates.factory import convert_to_anthropic_tool_invoke

    result = convert_to_anthropic_tool_invoke(
        tool_calls=[
            {
                "id": "srvtoolu_01Unpaired",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"query": "zig version"}'},
            }
        ],
        web_search_results=None,
        tool_results=None,
    )

    assert result == [
        {
            "type": "tool_use",
            "id": "srvtoolu_01Unpaired",
            "name": "web_search",
            "input": {"query": "zig version"},
        }
    ]


def test_convert_to_anthropic_tool_invoke_keeps_paired_server_tool_use():
    """When the paired server tool result is available, the srvtoolu_ call is
    still reconstructed as server_tool_use followed by its result block."""
    from litellm.litellm_core_utils.prompt_templates.factory import convert_to_anthropic_tool_invoke

    server_result = {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_01Paired",
        "content": [{"type": "web_search_result", "url": "https://ziglang.org", "title": "Zig"}],
    }

    result = convert_to_anthropic_tool_invoke(
        tool_calls=[
            {
                "id": "srvtoolu_01Paired",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"query": "zig version"}'},
            }
        ],
        web_search_results=[server_result],
        tool_results=None,
    )

    assert result == [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_01Paired",
            "name": "web_search",
            "input": {"query": "zig version"},
        },
        server_result,
    ]


def test_anthropic_messages_pt_keeps_system_role_after_user_turn():
    """Models flagged supports_mid_conversation_system accept role=system inside
    messages; the converter must emit it as a system message with its text
    blocks and cache_control intact instead of rejecting the role."""
    messages = [
        {"role": "user", "content": "First question"},
        {
            "role": "system",
            "content": [{"type": "text", "text": "Answer in one word.", "cache_control": {"type": "ephemeral"}}],
        },
        {"role": "assistant", "content": "Yes"},
        {"role": "user", "content": "Second question"},
    ]

    result = anthropic_messages_pt(messages=messages, model="claude-opus-4-8", llm_provider="anthropic")

    assert [m["role"] for m in result] == ["user", "system", "assistant", "user"]
    assert result[1] == {
        "role": "system",
        "content": [{"type": "text", "text": "Answer in one word.", "cache_control": {"type": "ephemeral"}}],
    }


def test_anthropic_messages_pt_system_string_content_becomes_text_block():
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "system", "content": "Answer in one word."},
    ]

    result = anthropic_messages_pt(messages=messages, model="claude-opus-4-8", llm_provider="anthropic")

    assert result[1] == {"role": "system", "content": [{"type": "text", "text": "Answer in one word."}]}


def test_anthropic_messages_pt_drops_a_system_message_with_no_text():
    """Anthropic rejects empty text blocks, so a text-less system message must
    vanish rather than reach the wire as an empty system turn."""
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "system", "content": ""},
        {"role": "assistant", "content": "Yes"},
    ]

    result = anthropic_messages_pt(messages=messages, model="claude-opus-4-8", llm_provider="anthropic")

    assert [m["role"] for m in result] == ["user", "assistant"]


def test_anthropic_messages_pt_drops_empty_but_signed_thinking_block():
    """
    Anthropic rejects a `thinking` block whose `thinking` text is empty, even
    when it carries a valid-looking signature, with:
        400 messages.N.content.M.thinking: each thinking block must contain thinking
    This shape is reachable via cross-provider replay of a `thinking_blocks`
    history item (see PR #36033), so `is_unsignable_thinking_block()` must
    also check the thinking text, not just the signature.
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        anthropic_messages_pt,
    )

    messages = [
        {"role": "user", "content": "What's 2+2?"},
        {
            "role": "assistant",
            "content": "4",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "",
                    "signature": "sig_abc123_looks_valid",
                }
            ],
        },
    ]

    result = anthropic_messages_pt(
        messages=messages,
        model="claude-sonnet-4-5-20250929",
        llm_provider="anthropic",
    )

    assistant_msg = result[1]
    assert isinstance(assistant_msg["content"], list)
    content_types = [block.get("type") for block in assistant_msg["content"]]
    assert "thinking" not in content_types, "empty-text thinking block must be dropped even though it has a signature"


def test_anthropic_messages_pt_keeps_non_empty_signed_thinking_block():
    """
    Regression: a real, non-empty, signed thinking block must still pass
    through unchanged.
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        anthropic_messages_pt,
    )

    messages = [
        {"role": "user", "content": "What's 2+2?"},
        {
            "role": "assistant",
            "content": "4",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Let me add these numbers together.",
                    "signature": "sig_abc123_looks_valid",
                }
            ],
        },
    ]

    result = anthropic_messages_pt(
        messages=messages,
        model="claude-sonnet-4-5-20250929",
        llm_provider="anthropic",
    )

    assistant_msg = result[1]
    assert isinstance(assistant_msg["content"], list)
    thinking_block = next((b for b in assistant_msg["content"] if b.get("type") == "thinking"), None)
    assert thinking_block is not None, "non-empty signed thinking block must be kept"
    assert thinking_block["thinking"] == "Let me add these numbers together."
    assert thinking_block["signature"] == "sig_abc123_looks_valid"


def test_anthropic_messages_pt_keeps_redacted_thinking_block():
    """
    Regression: `redacted_thinking` blocks carry no signature and no plaintext
    `thinking` field by design, and must always be kept regardless of the new
    emptiness check (which only applies to `type == "thinking"` blocks).
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        anthropic_messages_pt,
    )

    messages = [
        {"role": "user", "content": "What's 2+2?"},
        {
            "role": "assistant",
            "content": "4",
            "thinking_blocks": [
                {
                    "type": "redacted_thinking",
                    "data": "encrypted_opaque_blob",
                }
            ],
        },
    ]

    result = anthropic_messages_pt(
        messages=messages,
        model="claude-sonnet-4-5-20250929",
        llm_provider="anthropic",
    )

    assistant_msg = result[1]
    assert isinstance(assistant_msg["content"], list)
    content_types = [block.get("type") for block in assistant_msg["content"]]
    assert "redacted_thinking" in content_types, "redacted_thinking blocks must always be kept"


def test_anthropic_messages_pt_drops_unsigned_thinking_block():
    """
    Regression (pre-existing behaviour): a thinking block with no signature
    (or an empty/null one) must still be dropped, independent of whether the
    thinking text is populated.
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        anthropic_messages_pt,
    )

    messages = [
        {"role": "user", "content": "What's 2+2?"},
        {
            "role": "assistant",
            "content": "4",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Let me add these numbers together.",
                    "signature": "",
                }
            ],
        },
    ]

    result = anthropic_messages_pt(
        messages=messages,
        model="claude-sonnet-4-5-20250929",
        llm_provider="anthropic",
    )

    assistant_msg = result[1]
    assert isinstance(assistant_msg["content"], list)
    content_types = [block.get("type") for block in assistant_msg["content"]]
    assert "thinking" not in content_types, "unsigned thinking block must still be dropped"


def test_is_unsignable_thinking_block_treats_whitespace_only_as_empty():
    """
    Edge case: a `thinking` field that is present but whitespace-only (e.g.
    a single trailing newline forwarded from another provider's empty
    reasoning summary) is functionally empty and Anthropic's API will still
    reject it with "each thinking block must contain thinking". We treat it
    the same as a fully empty string and drop the block.

    The check lives in the shared `is_unsignable_thinking_block` helper, which
    `_drop_unsignable_thinking_blocks` calls standalone, so the whitespace-aware
    test has to hold there rather than only at the factory call site.
    """
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        is_unsignable_thinking_block,
    )

    whitespace_only_block = {
        "type": "thinking",
        "thinking": "   \n\t  ",
        "signature": "sig_abc123_looks_valid",
    }

    assert is_unsignable_thinking_block(whitespace_only_block) is True


_CONTENT_LESS_USER_MESSAGES: Final = ({"role": "user"}, {"role": "user", "content": None})
_CONTENT_LESS_TOOL_MESSAGES: Final = (
    {"role": "tool", "tool_call_id": "call_1"},
    {"role": "tool", "tool_call_id": "call_1", "content": None},
)
_BOSTON_WEATHER_TOOL_CALL_TURN: Final = (
    {"role": "user", "content": "What is the weather in Boston?"},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Boston"}'},
            }
        ],
    },
)


def _conversation_around(
    content_less_user_message: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    with_message: Final = [
        {"role": "user", "content": "What is the capital of France?"},
        content_less_user_message,
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "And of Spain?"},
    ]
    without_message: Final = [message for message in with_message if message is not content_less_user_message]
    return validate_and_fix_openai_messages(with_message), validate_and_fix_openai_messages(without_message)


@pytest.mark.parametrize("content_less_user_message", _CONTENT_LESS_USER_MESSAGES)
def test_bedrock_converse_messages_pt_user_message_without_content_adds_no_block(
    content_less_user_message: dict[str, object],
):
    with_message, without_message = _conversation_around(content_less_user_message)

    assert _bedrock_converse_messages_pt(
        messages=with_message, model="anthropic.claude-haiku-4-5", llm_provider="bedrock"
    ) == _bedrock_converse_messages_pt(
        messages=without_message, model="anthropic.claude-haiku-4-5", llm_provider="bedrock"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("content_less_user_message", _CONTENT_LESS_USER_MESSAGES)
async def test_bedrock_converse_messages_pt_async_user_message_without_content_adds_no_block(
    content_less_user_message: dict[str, object],
):
    with_message, without_message = _conversation_around(content_less_user_message)

    assert await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages=with_message, model="anthropic.claude-haiku-4-5", llm_provider="bedrock"
    ) == await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages=without_message, model="anthropic.claude-haiku-4-5", llm_provider="bedrock"
    )


@pytest.mark.parametrize("content_less_tool_message", _CONTENT_LESS_TOOL_MESSAGES)
def test_bedrock_converse_messages_pt_tool_message_without_content_yields_empty_tool_result(
    content_less_tool_message: dict[str, object],
):
    result: Final = _bedrock_converse_messages_pt(
        messages=validate_and_fix_openai_messages([*_BOSTON_WEATHER_TOOL_CALL_TURN, content_less_tool_message]),
        model="anthropic.claude-haiku-4-5",
        llm_provider="bedrock",
    )

    tool_result: Final = result[-1]["content"][0]["toolResult"]
    assert result[-1]["role"] == "user"
    assert tool_result["toolUseId"] == "call_1"
    assert tool_result["content"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("content_less_tool_message", _CONTENT_LESS_TOOL_MESSAGES)
async def test_bedrock_converse_messages_pt_async_tool_message_without_content_yields_empty_tool_result(
    content_less_tool_message: dict[str, object],
):
    result: Final = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages=validate_and_fix_openai_messages([*_BOSTON_WEATHER_TOOL_CALL_TURN, content_less_tool_message]),
        model="anthropic.claude-haiku-4-5",
        llm_provider="bedrock",
    )

    tool_result: Final = result[-1]["content"][0]["toolResult"]
    assert tool_result["toolUseId"] == "call_1"
    assert tool_result["content"] == []


def test_bedrock_converse_messages_pt_blank_user_text_sends_the_continue_message_text():
    continue_message: Final = {"role": "user", "content": "Please continue."}
    blank_last_turn: Final = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi."},
        {"role": "user", "content": "   "},
    ]
    explicit_last_turn: Final = [*blank_last_turn[:2], continue_message]

    assert _bedrock_converse_messages_pt(
        messages=blank_last_turn,
        model="anthropic.claude-haiku-4-5",
        llm_provider="bedrock",
        user_continue_message=continue_message,
    ) == _bedrock_converse_messages_pt(
        messages=explicit_last_turn,
        model="anthropic.claude-haiku-4-5",
        llm_provider="bedrock",
        user_continue_message=continue_message,
    )


@pytest.mark.parametrize("content_less_user_message", _CONTENT_LESS_USER_MESSAGES)
def test_bedrock_converse_messages_pt_lone_content_less_user_turn_sends_the_continue_message(
    content_less_user_message: dict[str, object],
):
    continue_message: Final = {"role": "user", "content": "Please continue."}

    assert _bedrock_converse_messages_pt(
        messages=validate_and_fix_openai_messages([content_less_user_message]),
        model="anthropic.claude-haiku-4-5",
        llm_provider="bedrock",
        user_continue_message=continue_message,
    ) == _bedrock_converse_messages_pt(
        messages=[continue_message],
        model="anthropic.claude-haiku-4-5",
        llm_provider="bedrock",
        user_continue_message=continue_message,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("content_less_user_message", _CONTENT_LESS_USER_MESSAGES)
async def test_bedrock_converse_messages_pt_async_lone_content_less_user_turn_continues_under_modify_params(
    content_less_user_message: dict[str, object], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "modify_params", True)

    assert await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages=validate_and_fix_openai_messages([content_less_user_message]),
        model="anthropic.claude-haiku-4-5",
        llm_provider="bedrock",
    ) == await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages=[{"role": "user", "content": ""}],
        model="anthropic.claude-haiku-4-5",
        llm_provider="bedrock",
    )


@pytest.mark.parametrize("content_less_user_message", _CONTENT_LESS_USER_MESSAGES)
def test_bedrock_converse_messages_pt_lone_content_less_user_turn_adds_no_block_without_a_continue_message(
    content_less_user_message: dict[str, object], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "modify_params", False)

    assert (
        _bedrock_converse_messages_pt(
            messages=validate_and_fix_openai_messages([content_less_user_message]),
            model="anthropic.claude-haiku-4-5",
            llm_provider="bedrock",
        )
        == []
    )


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

@pytest.fixture(scope="function")
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Resets litellm globals to their true defaults before each test and
    restores them afterward, so tests don't leak side effects.
    Works safely under pytest-xdist parallel execution.
    """
    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in ("pre_call_rules", "post_call_rules"):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
        "pre_call_rules",
        "post_call_rules",
    ):
        if hasattr(litellm, attr):
            setattr(litellm, attr, [])
    for attr, default_val in _SCALAR_DEFAULTS.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, default_val)
    yield
    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    _invalidate_model_cost_lowercase_map()

_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "num_retries_per_request": getattr(litellm, "num_retries_per_request", None),
    "request_timeout": getattr(litellm, "request_timeout", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "default_fallbacks": getattr(litellm, "default_fallbacks", None),
    "enable_azure_ad_token_refresh": getattr(litellm, "enable_azure_ad_token_refresh", None),
    "tag_budget_config": getattr(litellm, "tag_budget_config", None),
    "model_cost": getattr(litellm, "model_cost", None),
    "token_counter": getattr(litellm, "token_counter", None),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
}

@pytest.fixture(scope="module")
def setup_and_teardown():
    """
    Module-scoped setup. Reloads litellm only in single-process mode
    (skipped under xdist to avoid cross-worker interference).
    """
    import litellm

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker_id is None:
        importlib.reload(litellm)
        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server

                importlib.reload(litellm.proxy.proxy_server)
        except Exception as e:
            print(f"Error reloading litellm.proxy.proxy_server: {e}")
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()
    yield

@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
def test_function_call_non_openai_model():
    try:
        model = "claude-3-5-haiku-20241022"
        messages = [{"role": "user", "content": "what's the weather in sf?"}]
        functions = [
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        ]
        response = litellm.completion(model=model, messages=messages, functions=functions)
        pytest.fail(f"An error occurred")
    except Exception as e:
        print(e)
        pass

@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
def test_empty_content():
    """
    Make a chat completions request with empty content -> expect this to work
    """
    rules_obj = Rules()

    def completion():
        pass

    function_setup(
        original_function="completion",
        rules_obj=rules_obj,
        start_time=datetime.now(),
        messages=[],
        litellm_call_id=str(uuid.uuid4()),
    )

@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
def test_thought_signature_removal_for_non_gemini():
    """
    Test that thought signatures are removed from tool call IDs when sending to non-Gemini models
    """
    rules_obj = Rules()

    # Create messages with thought signatures (as would come from Gemini)
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_123{THOUGHT_SIGNATURE_SEPARATOR}sig1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "SF"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call_123{THOUGHT_SIGNATURE_SEPARATOR}sig1",
            "content": "Sunny, 72°F",
        },
    ]

    # Call function_setup with OpenAI model (non-Gemini)
    logging_obj, kwargs = function_setup(
        original_function="acompletion",
        rules_obj=rules_obj,
        start_time=datetime.now(),
        model="gpt-4",
        messages=messages,
        litellm_call_id=str(uuid.uuid4()),
        custom_llm_provider="openai",
    )

    # Verify thought signatures were removed
    processed_messages = kwargs["messages"]
    assert processed_messages[1]["tool_calls"][0]["id"] == "call_123"
    assert processed_messages[2]["tool_call_id"] == "call_123"
    assert THOUGHT_SIGNATURE_SEPARATOR not in processed_messages[1]["tool_calls"][0]["id"]
    assert THOUGHT_SIGNATURE_SEPARATOR not in processed_messages[2]["tool_call_id"]

@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
def test_thought_signature_preserved_for_gemini():
    """
    Test that thought signatures are preserved when sending to Gemini models
    """
    rules_obj = Rules()

    # Create messages with thought signatures
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_456{THOUGHT_SIGNATURE_SEPARATOR}sig2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call_456{THOUGHT_SIGNATURE_SEPARATOR}sig2",
            "content": "Rainy, 65°F",
        },
    ]

    # Call function_setup with Gemini model
    logging_obj, kwargs = function_setup(
        original_function="acompletion",
        rules_obj=rules_obj,
        start_time=datetime.now(),
        model="gemini-1.5-pro",
        messages=messages,
        litellm_call_id=str(uuid.uuid4()),
        custom_llm_provider="vertex_ai",
    )

    # Verify thought signatures were preserved (messages should be unchanged)
    processed_messages = kwargs["messages"]
    assert THOUGHT_SIGNATURE_SEPARATOR in processed_messages[1]["tool_calls"][0]["id"]
    assert THOUGHT_SIGNATURE_SEPARATOR in processed_messages[2]["tool_call_id"]

@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
def test_thought_signature_removal_with_multiple_tool_calls():
    """
    Test that thought signatures are removed from multiple tool calls
    """
    rules_obj = Rules()

    messages = [
        {"role": "user", "content": "Get weather and time"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_1{THOUGHT_SIGNATURE_SEPARATOR}sig1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                },
                {
                    "id": f"call_2{THOUGHT_SIGNATURE_SEPARATOR}sig2",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call_1{THOUGHT_SIGNATURE_SEPARATOR}sig1",
            "content": "Sunny",
        },
        {
            "role": "tool",
            "tool_call_id": f"call_2{THOUGHT_SIGNATURE_SEPARATOR}sig2",
            "content": "3:00 PM",
        },
    ]

    logging_obj, kwargs = function_setup(
        original_function="acompletion",
        rules_obj=rules_obj,
        start_time=datetime.now(),
        model="claude-3-opus",
        messages=messages,
        litellm_call_id=str(uuid.uuid4()),
        custom_llm_provider="anthropic",
    )

    processed_messages = kwargs["messages"]

    # Check all tool call IDs are cleaned
    assert processed_messages[1]["tool_calls"][0]["id"] == "call_1"
    assert processed_messages[1]["tool_calls"][1]["id"] == "call_2"
    assert processed_messages[2]["tool_call_id"] == "call_1"
    assert processed_messages[3]["tool_call_id"] == "call_2"

@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
def test_messages_without_tool_calls_unchanged():
    """
    Test that messages without tool calls pass through unchanged
    """
    rules_obj = Rules()

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    logging_obj, kwargs = function_setup(
        original_function="acompletion",
        rules_obj=rules_obj,
        start_time=datetime.now(),
        model="gpt-4",
        messages=messages,
        litellm_call_id=str(uuid.uuid4()),
        custom_llm_provider="openai",
    )

    # Messages should be unchanged
    assert kwargs["messages"] == messages


def test_llama_3_prompt():
    messages = [
        {"role": "system", "content": "You are a good bot"},
        {"role": "user", "content": "Hey, how's it going?"},
    ]
    received_prompt = prompt_factory(model="meta-llama/Meta-Llama-3-8B-Instruct", messages=messages)
    print(f"received_prompt: {received_prompt}")

    expected_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a good bot<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHey, how's it going?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
    assert received_prompt == expected_prompt


def test_codellama_prompt_format():
    messages = [
        {"role": "system", "content": "You are a good bot"},
        {"role": "user", "content": "Hey, how's it going?"},
    ]
    expected_prompt = "<s>[INST] <<SYS>>\nYou are a good bot\n<</SYS>>\n [/INST]\n[INST] Hey, how's it going? [/INST]\n"
    assert llama_2_chat_pt(messages) == expected_prompt


def test_claude_2_1_pt_formatting():

    messages = [{"role": "user", "content": "Hello"}]
    expected_prompt = "\n\nHuman: Hello\n\nAssistant: "
    assert claude_2_1_pt(messages) == expected_prompt

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": 'Please return "Hello World" as a JSON object.'},
        {"role": "assistant", "content": "{"},
    ]
    expected_prompt = (
        'You are a helpful assistant.\n\nHuman: Please return "Hello World" as a JSON object.\n\nAssistant: {'
    )
    assert claude_2_1_pt(messages) == expected_prompt

    messages = [
        {"role": "system", "content": "You are a storyteller."},
        {"role": "assistant", "content": "Once upon a time, there "},
    ]
    expected_prompt = "You are a storyteller.\n\nHuman: \n\nAssistant: Once upon a time, there "
    assert claude_2_1_pt(messages) == expected_prompt

    messages = [
        {"role": "system", "content": "System reboot"},
        {"role": "user", "content": "Is everything okay?"},
    ]
    expected_prompt = "System reboot\n\nHuman: Is everything okay?\n\nAssistant: "
    assert claude_2_1_pt(messages) == expected_prompt


def test_anthropic_pt_formatting():

    messages = [{"role": "user", "content": "Hello"}]
    expected_prompt = "\n\nHuman: Hello\n\nAssistant: "
    assert anthropic_pt(messages) == expected_prompt

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": 'Please return "Hello World" as a JSON object.'},
        {"role": "assistant", "content": "{"},
    ]
    expected_prompt = '\n\nHuman: <admin>You are a helpful assistant.</admin>\n\nHuman: Please return "Hello World" as a JSON object.\n\nAssistant: {'
    assert anthropic_pt(messages) == expected_prompt

    messages = [
        {"role": "system", "content": "You are a storyteller."},
        {"role": "assistant", "content": "Once upon a time, there "},
    ]
    expected_prompt = "\n\nHuman: <admin>You are a storyteller.</admin>\n\nAssistant: Once upon a time, there "
    assert anthropic_pt(messages) == expected_prompt

    messages = [
        {"role": "system", "content": "System reboot"},
        {"role": "user", "content": "Is everything okay?"},
    ]
    expected_prompt = "\n\nHuman: <admin>System reboot</admin>\n\nHuman: Is everything okay?\n\nAssistant: "
    assert anthropic_pt(messages) == expected_prompt


def test_anthropic_messages_nested_pt():

    messages = [
        {"content": [{"text": "here is a task", "type": "text"}], "role": "user"},
        {
            "content": [{"text": "sure happy to help", "type": "text"}],
            "role": "assistant",
        },
        {
            "content": [
                {
                    "text": "Here is a screenshot of the current desktop with the "
                    "mouse coordinates (500, 350). Please select an action "
                    "from the provided schema.",
                    "type": "text",
                }
            ],
            "role": "user",
        },
    ]

    new_messages = anthropic_messages_pt(messages, model="claude-3-sonnet-20240229", llm_provider="anthropic")

    assert isinstance(new_messages[1]["content"][0]["text"], str)


def test_bedrock_tool_calling_pt():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    converted_tools = _bedrock_tools_pt(tools=tools)

    print(converted_tools)
    assert converted_tools[0]["toolSpec"]["name"] == "get_current_weather" and (
        converted_tools[0]["toolSpec"]["inputSchema"]["json"] == tools[0]["function"]["parameters"]
    )


@pytest.mark.parametrize(
    "url, expected_media_type",
    [
        ("data:image/jpeg;base64,1234", "image/jpeg"),
        ("data:application/pdf;base64,1234", "application/pdf"),
        (r"data:image\/jpeg;base64,1234", "image/jpeg"),
    ],
)
def test_base64_image_input(url, expected_media_type):
    response = convert_to_anthropic_image_obj(openai_image_url=url, format=None)

    assert response["media_type"] == expected_media_type


def test_create_anthropic_image_param_with_http_url():
    """Test that HTTP/HTTPS URLs are passed as URL references, not base64."""
    image_param = create_anthropic_image_param("https://example.com/image.jpg", format=None)

    assert image_param["type"] == "image"
    assert image_param["source"]["type"] == "url"
    assert image_param["source"]["url"] == "https://example.com/image.jpg"


def test_create_anthropic_image_param_with_https_url():
    """Test that HTTPS URLs are passed as URL references."""
    image_param = create_anthropic_image_param("https://example.com/image.png", format=None)

    assert image_param["type"] == "image"
    assert image_param["source"]["type"] == "url"
    assert image_param["source"]["url"] == "https://example.com/image.png"


def test_create_anthropic_image_param_with_dict_input():
    """Test that dict input with URL is handled correctly."""
    image_param = create_anthropic_image_param(
        {"url": "https://example.com/image.jpg", "format": "image/jpeg"}, format=None
    )

    assert image_param["type"] == "image"
    assert image_param["source"]["type"] == "url"
    assert image_param["source"]["url"] == "https://example.com/image.jpg"


def test_create_anthropic_image_param_with_base64_data_uri():
    """Test that data URIs are converted to base64."""
    image_param = create_anthropic_image_param("data:image/jpeg;base64,/9j/4AAQSkZJRg==", format=None)

    assert image_param["type"] == "image"
    assert image_param["source"]["type"] == "base64"
    assert image_param["source"]["media_type"] == "image/jpeg"
    assert image_param["source"]["data"] == "/9j/4AAQSkZJRg=="


def test_create_anthropic_image_param_with_format_override():
    """Test that format parameter can override media type."""
    image_param = create_anthropic_image_param("data:image/jpeg;base64,1234", format="image/png")

    assert image_param["type"] == "image"
    assert image_param["source"]["type"] == "base64"
    assert image_param["source"]["media_type"] == "image/png"


def test_anthropic_messages_pt_with_url_image():
    """Test that anthropic_messages_pt correctly handles HTTP/HTTPS URLs as URL references."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": "https://example.com/image.jpg",
                },
            ],
        }
    ]

    result = anthropic_messages_pt(messages=messages, model="claude-3-5-sonnet", llm_provider="anthropic")

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], list)
    assert len(result[0]["content"]) == 2

    assert result[0]["content"][0]["type"] == "text"

    assert result[0]["content"][1]["type"] == "image"
    assert result[0]["content"][1]["source"]["type"] == "url"
    assert result[0]["content"][1]["source"]["url"] == "https://example.com/image.jpg"


def test_anthropic_messages_pt_with_base64_image():
    """Test that anthropic_messages_pt correctly handles data URIs as base64."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
                },
            ],
        }
    ]

    result = anthropic_messages_pt(messages=messages, model="claude-3-5-sonnet", llm_provider="anthropic")

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], list)
    assert len(result[0]["content"]) == 2

    assert result[0]["content"][1]["type"] == "image"
    assert result[0]["content"][1]["source"]["type"] == "base64"
    assert result[0]["content"][1]["source"]["media_type"] == "image/jpeg"


def test_anthropic_messages_tool_call():
    messages = [
        {
            "role": "user",
            "content": "Would development of a software platform be under ASC 350-40 or ASC 985?",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_call_id": "bc8cb4b6-88c4-4138-8993-3a9d9cd51656",
            "tool_calls": [
                {
                    "id": "bc8cb4b6-88c4-4138-8993-3a9d9cd51656",
                    "function": {
                        "arguments": '{"completed_steps": [], "next_steps": [{"tool_name": "AccountingResearchTool", "description": "Research ASC 350-40 to understand its scope and applicability to software development."}, {"tool_name": "AccountingResearchTool", "description": "Research ASC 985 to understand its scope and applicability to software development."}, {"tool_name": "AccountingResearchTool", "description": "Compare the scopes of ASC 350-40 and ASC 985 to determine which is more applicable to software platform development."}], "learnings": [], "potential_issues": ["The distinction between the two standards might not be clear-cut for all types of software development.", "There might be specific circumstances or details about the software platform that could affect which standard applies."], "missing_info": ["Specific details about the type of software platform being developed (e.g., for internal use or for sale).", "Whether the entity developing the software is also the end-user or if it\'s being developed for external customers."], "done": false, "required_formatting": null}',
                        "name": "TaskPlanningTool",
                    },
                    "type": "function",
                }
            ],
        },
        {
            "role": "function",
            "content": '{"completed_steps":[],"next_steps":[{"tool_name":"AccountingResearchTool","description":"Research ASC 350-40 to understand its scope and applicability to software development."},{"tool_name":"AccountingResearchTool","description":"Research ASC 985 to understand its scope and applicability to software development."},{"tool_name":"AccountingResearchTool","description":"Compare the scopes of ASC 350-40 and ASC 985 to determine which is more applicable to software platform development."}],"formatting_step":null}',
            "name": "TaskPlanningTool",
            "tool_call_id": "bc8cb4b6-88c4-4138-8993-3a9d9cd51656",
        },
    ]

    translated_messages = anthropic_messages_pt(messages, model="claude-3-sonnet-20240229", llm_provider="anthropic")

    print(translated_messages)

    assert translated_messages[-1]["content"][0]["tool_use_id"] == "bc8cb4b6-88c4-4138-8993-3a9d9cd51656"


def test_anthropic_cache_controls_pt():
    "see anthropic docs for this: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#continuing-a-multi-turn-conversation"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    translated_messages = anthropic_messages_pt(messages, model="claude-3-5-sonnet-20240620", llm_provider="anthropic")

    for i, msg in enumerate(translated_messages):
        if i == 0:
            assert msg["content"][0]["cache_control"] == {"type": "ephemeral"}
        elif i == 1:
            assert "cache_controls" not in msg["content"][0]
        elif i == 2:
            assert msg["content"][0]["cache_control"] == {"type": "ephemeral"}
        elif i == 3:
            assert msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    print("translated_messages: ", translated_messages)


def test_anthropic_cache_controls_tool_calls_pt():
    """
    Tests that cache_control is properly set in tool_calls when converting messages
    for the Anthropic API.
    """
    messages = [
        {
            "role": "user",
            "content": "Can you help me get the weather?",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "weather-tool-id-123",
                    "function": {
                        "arguments": '{"location": "San Francisco"}',
                        "name": "get_weather",
                    },
                    "type": "function",
                }
            ],
            "cache_control": {"type": "ephemeral"},
        },
        {
            "role": "function",
            "content": '{"temperature": 72, "unit": "fahrenheit", "description": "sunny"}',
            "name": "get_weather",
            "tool_call_id": "weather-tool-id-123",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    translated_messages = anthropic_messages_pt(messages, model="claude-3-sonnet-20240229", llm_provider="anthropic")

    print("Translated tool call messages:", translated_messages)

    assert translated_messages[0]["role"] == "user"

    assert translated_messages[1]["role"] == "assistant"
    for content_item in translated_messages[1]["content"]:
        if content_item["type"] == "tool_use":
            assert "cache_control" not in content_item
            assert content_item["name"] == "get_weather"

    assert translated_messages[2]["role"] == "user"
    for content_item in translated_messages[2]["content"]:
        if content_item["type"] == "tool_result":
            assert content_item["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize("provider", ["bedrock", "anthropic"])
def test_bedrock_parallel_tool_calling_pt(provider):
    """
    Make sure parallel tool call blocks are merged correctly - https://github.com/BerriAI/litellm/issues/5277
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        _bedrock_converse_messages_pt,
    )
    from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message

    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco, Tokyo, and Paris? - give me 3 responses",
        },
        Message(
            content="Here are the current weather conditions for San Francisco, Tokyo, and Paris:",
            role="assistant",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    index=1,
                    function=Function(
                        arguments='{"city": "New York"}',
                        name="get_current_weather",
                    ),
                    id="tooluse_XcqEBfm8R-2YVaPhDUHsPQ",
                    type="function",
                ),
                ChatCompletionMessageToolCall(
                    index=2,
                    function=Function(
                        arguments='{"city": "London"}',
                        name="get_current_weather",
                    ),
                    id="tooluse_VB9nk7UGRniVzGcaj6xrAQ",
                    type="function",
                ),
            ],
            function_call=None,
        ),
        {
            "tool_call_id": "tooluse_XcqEBfm8R-2YVaPhDUHsPQ",
            "role": "tool",
            "name": "get_current_weather",
            "content": "25 degrees celsius.",
        },
        {
            "tool_call_id": "tooluse_VB9nk7UGRniVzGcaj6xrAQ",
            "role": "tool",
            "name": "get_current_weather",
            "content": "28 degrees celsius.",
        },
    ]

    if provider == "bedrock":
        translated_messages = _bedrock_converse_messages_pt(
            messages=messages,
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            llm_provider="bedrock",
        )
    else:
        translated_messages = anthropic_messages_pt(
            messages=messages,
            model="claude-3-sonnet-20240229-v1:0",
            llm_provider=provider,
        )
    print(translated_messages)

    number_of_messages = len(translated_messages)

    assert translated_messages[number_of_messages - 1]["role"] != translated_messages[number_of_messages - 2]["role"]


def test_vertex_only_image_user_message():
    base64_image = "/9j/2wCEAAgGBgcGBQ"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        },
    ]

    response = gemini_convert_messages_with_history(messages=messages, model="gemini-1.5-pro")

    expected_response = [
        {
            "role": "user",
            "parts": [
                {
                    "inline_data": {
                        "data": "/9j/2wCEAAgGBgcGBQ",
                        "mime_type": "image/jpeg",
                    }
                },
                {"text": " "},
            ],
        }
    ]

    assert len(response) == len(expected_response)
    for idx, content in enumerate(response):
        assert content == expected_response[idx], "Invalid gemini input. Got={}, Expected={}".format(
            content, expected_response[idx]
        )


def test_no_messages_yields_user_text():
    """
    Test that contents are not empty and have text when called without messages
    This is to support blha blah
    """
    messages: List[AllMessageValues] = []

    contents = gemini_convert_messages_with_history(messages=messages)

    expected_output = [{"role": "user", "parts": [{"text": " "}]}]

    assert contents == expected_output


def test_convert_url(monkeypatch):
    import base64
    from unittest.mock import MagicMock

    import httpx

    from litellm.litellm_core_utils.prompt_templates.image_handling import (
        in_memory_cache,
    )

    url = "https://picsum.photos/id/237/200/300"
    image_bytes = b"\x89PNG\r\n\x1a\nfake-png-bytes"

    mock_client = MagicMock()
    mock_client.get.return_value = httpx.Response(200, content=image_bytes, headers={"Content-Type": "image/png"})

    monkeypatch.setattr(litellm, "user_url_validation", False, raising=False)
    monkeypatch.setattr(litellm, "module_level_client", mock_client, raising=False)
    in_memory_cache.flush_cache()

    result = convert_url_to_base64(url)

    expected = "data:image/png;base64," + base64.b64encode(image_bytes).decode("utf-8")
    assert result == expected
    mock_client.get.assert_called_once()


def test_azure_tool_call_invoke_helper():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Copenhagen?"},
        {"role": "assistant", "function_call": {"name": "get_weather"}},
    ]

    transformed_messages = litellm.AzureOpenAIConfig().transform_request(
        model="gpt-4o",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert transformed_messages["messages"] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Copenhagen?"},
        {
            "role": "assistant",
            "function_call": {"name": "get_weather", "arguments": ""},
        },
    ]


@pytest.mark.parametrize(
    "messages, expected_messages, user_continue_message, assistant_continue_message",
    [
        (
            [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hello! How can I assist you today?"},
                {"role": "user", "content": "What is Databricks?"},
                {"role": "user", "content": "What is Azure?"},
                {"role": "assistant", "content": "I don't know anyything, do you?"},
            ],
            [
                {"role": "user", "content": "Hello!"},
                {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?",
                },
                {"role": "user", "content": "What is Databricks?"},
                {
                    "role": "assistant",
                    "content": "Please continue.",
                },
                {"role": "user", "content": "What is Azure?"},
                {
                    "role": "assistant",
                    "content": "I don't know anyything, do you?",
                },
                {
                    "role": "user",
                    "content": "Please continue.",
                },
            ],
            None,
            None,
        ),
        (
            [
                {"role": "user", "content": "Hello!"},
            ],
            [
                {"role": "user", "content": "Hello!"},
            ],
            None,
            None,
        ),
        (
            [
                {"role": "user", "content": "Hello!"},
                {"role": "user", "content": "What is Databricks?"},
            ],
            [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Please continue."},
                {"role": "user", "content": "What is Databricks?"},
            ],
            None,
            None,
        ),
        (
            [
                {"role": "user", "content": "Hello!"},
                {"role": "user", "content": "What is Databricks?"},
                {"role": "user", "content": "What is Azure?"},
            ],
            [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Please continue."},
                {"role": "user", "content": "What is Databricks?"},
                {
                    "role": "assistant",
                    "content": "Please continue.",
                },
                {"role": "user", "content": "What is Azure?"},
            ],
            None,
            None,
        ),
        (
            [
                {"role": "user", "content": "Hello!"},
                {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?",
                },
                {"role": "user", "content": "What is Databricks?"},
                {"role": "user", "content": "What is Azure?"},
                {"role": "assistant", "content": "I don't know anyything, do you?"},
                {"role": "assistant", "content": "I can't repeat sentences."},
            ],
            [
                {"role": "user", "content": "Hello!"},
                {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?",
                },
                {"role": "user", "content": "What is Databricks?"},
                {
                    "role": "assistant",
                    "content": "Please continue",
                },
                {"role": "user", "content": "What is Azure?"},
                {
                    "role": "assistant",
                    "content": "I don't know anyything, do you?",
                },
                {
                    "role": "user",
                    "content": "Ok",
                },
                {
                    "role": "assistant",
                    "content": "I can't repeat sentences.",
                },
                {"role": "user", "content": "Ok"},
            ],
            {
                "role": "user",
                "content": "Ok",
            },
            {
                "role": "assistant",
                "content": "Please continue",
            },
        ),
    ],
)
def test_ensure_alternating_roles(messages, expected_messages, user_continue_message, assistant_continue_message):
    messages = get_completion_messages(
        messages=messages,
        assistant_continue_message=assistant_continue_message,
        user_continue_message=user_continue_message,
        ensure_alternating_roles=True,
    )

    print(messages)

    assert messages == expected_messages


def test_ensure_alternating_roles_with_tool_calls():
    """Fixes Regression in #18685"""
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "72F, sunny"},
        {"role": "assistant", "content": "It's 72F and sunny in NYC."},
        {"role": "user", "content": "What about tomorrow?"},
        {"role": "user", "content": "And the day after?"},
        {"role": "user", "content": "What about next week?"},
    ]

    transformed_messages = get_completion_messages(
        messages=messages,
        assistant_continue_message=None,
        user_continue_message=None,
        ensure_alternating_roles=True,
    )

    assert transformed_messages == [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "72F, sunny"},
        {"role": "assistant", "content": "It's 72F and sunny in NYC."},
        {"role": "user", "content": "What about tomorrow?"},
        {"role": "assistant", "content": "Please continue."},
        {"role": "user", "content": "And the day after?"},
        {"role": "assistant", "content": "Please continue."},
        {"role": "user", "content": "What about next week?"},
    ]


def test_ensure_alternating_roles_three_consecutive_assistants():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "A1"},
        {"role": "assistant", "content": "A2"},
        {"role": "assistant", "content": "A3"},
    ]

    transformed_messages = get_completion_messages(
        messages=messages,
        assistant_continue_message=None,
        user_continue_message=None,
        ensure_alternating_roles=True,
    )

    assert transformed_messages == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Please continue."},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "Please continue."},
        {"role": "assistant", "content": "A3"},
        {"role": "user", "content": "Please continue."},
    ]


def test_ensure_alternating_roles_inserts_assistant_continue_across_tool_chain():
    """[user, assistant(tc), tool, user] gets assistant_continue before the second user."""
    messages = [
        {"role": "user", "content": "Search for X"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "results"},
        {"role": "user", "content": "Thanks, now do Y"},
    ]

    transformed_messages = get_completion_messages(
        messages=messages,
        assistant_continue_message=None,
        user_continue_message=None,
        ensure_alternating_roles=True,
    )

    assert transformed_messages == [
        {"role": "user", "content": "Search for X"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "results"},
        {"role": "assistant", "content": "Please continue."},
        {"role": "user", "content": "Thanks, now do Y"},
    ]


def test_ensure_alternating_roles_assistant_tool_call_then_assistant():
    """
    Malformed [assistant(tc), assistant(no-tc), user]:
    user_continue inserts break between adjacents, then assistant_continue
    fills the counted-sequence gap.
    """
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {"role": "assistant", "content": "Here's what I found."},
        {"role": "user", "content": "Thanks"},
    ]

    transformed_messages = get_completion_messages(
        messages=messages,
        assistant_continue_message=None,
        user_continue_message=None,
        ensure_alternating_roles=True,
    )

    assert transformed_messages == [
        {"role": "user", "content": "Please continue."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {"role": "assistant", "content": "Please continue."},
        {"role": "user", "content": "Please continue."},
        {"role": "assistant", "content": "Here's what I found."},
        {"role": "user", "content": "Thanks"},
    ]


def test_ensure_alternating_roles_trailing_tool_call_assistant():
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}',
                    },
                }
            ],
        },
    ]

    transformed_messages = get_completion_messages(
        messages=messages,
        assistant_continue_message=None,
        user_continue_message=None,
        ensure_alternating_roles=True,
    )

    assert transformed_messages == [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}',
                    },
                }
            ],
        },
        {"role": "assistant", "content": "Please continue."},
        {"role": "user", "content": "Please continue."},
    ]


def test_ensure_alternating_roles_multiple_tool_results():
    """[user, assistant(tc), tool, tool, user] — multiple tool results before next user."""
    messages = [
        {"role": "user", "content": "Search for X and Y"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search_x", "arguments": "{}"},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "search_y", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result X"},
        {"role": "tool", "tool_call_id": "c2", "content": "result Y"},
        {"role": "user", "content": "Thanks"},
    ]

    transformed_messages = get_completion_messages(
        messages=messages,
        assistant_continue_message=None,
        user_continue_message=None,
        ensure_alternating_roles=True,
    )

    assert transformed_messages == [
        {"role": "user", "content": "Search for X and Y"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search_x", "arguments": "{}"},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "search_y", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result X"},
        {"role": "tool", "tool_call_id": "c2", "content": "result Y"},
        {"role": "assistant", "content": "Please continue."},
        {"role": "user", "content": "Thanks"},
    ]


def test_ensure_alternating_roles_chained_tool_calls():
    """[user, assistant(tc), tool, assistant(tc), tool, user] — chained tool calls."""
    messages = [
        {"role": "user", "content": "Do multi-step task"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "step1", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "step1 done"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "step2", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "step2 done"},
        {"role": "user", "content": "What happened?"},
    ]

    transformed_messages = get_completion_messages(
        messages=messages,
        assistant_continue_message=None,
        user_continue_message=None,
        ensure_alternating_roles=True,
    )

    assert transformed_messages == [
        {"role": "user", "content": "Do multi-step task"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "step1", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "step1 done"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "step2", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "step2 done"},
        {"role": "assistant", "content": "Please continue."},
        {"role": "user", "content": "What happened?"},
    ]


def test_ensure_alternating_roles_system_prefix_with_tool_chain():
    """[system, user, assistant(tc), tool, user] — system prefix doesn't interfere."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Search for X"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "results"},
        {"role": "user", "content": "Thanks"},
    ]

    transformed_messages = get_completion_messages(
        messages=messages,
        assistant_continue_message=None,
        user_continue_message=None,
        ensure_alternating_roles=True,
    )

    assert transformed_messages == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Search for X"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "results"},
        {"role": "assistant", "content": "Please continue."},
        {"role": "user", "content": "Thanks"},
    ]


def test_just_system_message():
    from litellm.litellm_core_utils.prompt_templates.factory import (
        _bedrock_converse_messages_pt,
    )

    with pytest.raises(litellm.BadRequestError) as e:
        _bedrock_converse_messages_pt(
            messages=[],
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            llm_provider="bedrock",
        )

    assert "bedrock requires at least one non-system message" in str(e.value)


def test_hf_chat_template():
    from litellm.litellm_core_utils.prompt_templates.factory import (
        hf_chat_template,
    )

    model = "llama/arn:aws:bedrock:us-east-1:1234:imported-model/45d34re"
    litellm.register_prompt_template(
        model=model,
        tokenizer_config={
            "add_bos_token": True,
            "add_eos_token": False,
            "bos_token": {
                "__type": "AddedToken",
                "content": "",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False,
            },
            "clean_up_tokenization_spaces": False,
            "eos_token": {
                "__type": "AddedToken",
                "content": "",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False,
            },
            "legacy": True,
            "model_max_length": 16384,
            "pad_token": {
                "__type": "AddedToken",
                "content": "",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False,
            },
            "sp_model_kwargs": {},
            "unk_token": None,
            "tokenizer_class": "LlamaTokenizerFast",
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{' ' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{' ' + tool['type'] + ' ' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + ' '}}{%- set ns.is_first = true -%}{%- else %}{{' ' + tool['type'] + ' ' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + ' '}}{{' '}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{' ' + message['content'] + ' '}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{' ' + content + ' '}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{' ' + message['content'] + ' '}}{%- set ns.is_output_first = false %}{%- else %}{{' ' + message['content'] + ' '}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{' '}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{' '}}{% endif %}",
        },
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Copenhagen?"},
    ]
    chat_template = hf_chat_template(model=model, messages=messages)
    print(chat_template)
    assert chat_template.rstrip() == "You are a helpful assistant. What is the weather in Copenhagen?"


def test_ollama_pt():
    from litellm.litellm_core_utils.prompt_templates.factory import ollama_pt

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    prompt = ollama_pt(model="ollama/llama3.1", messages=messages)
    print(prompt)
    assert "You are a helpful assistant." in prompt["prompt"] and "Hello!" in prompt["prompt"]


def test_convert_to_anthropic_tool_invoke_regular_tool():
    """Test that regular tool_use is converted correctly."""
    tool_calls = [
        {
            "id": "toolu_01ABC123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}',
            },
        }
    ]

    result = convert_to_anthropic_tool_invoke(tool_calls)

    assert len(result) == 1
    assert result[0]["type"] == "tool_use"
    assert result[0]["id"] == "toolu_01ABC123"
    assert result[0]["name"] == "get_weather"
    assert result[0]["input"] == {"location": "San Francisco"}


def test_convert_to_anthropic_tool_invoke_sanitizes_invalid_ids():
    """Test that tool_use IDs with invalid characters are sanitized.

    Anthropic requires tool_use_id to match ^[a-zA-Z0-9_-]+$.
    IDs from external frameworks (e.g. MiniMax) may contain characters
    like colons that violate this pattern.
    """
    tool_calls = [
        {
            "id": "sessions_history:183",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Boston"}',
            },
        },
        {
            "id": "composio.NOTION_SEARCH",
            "type": "function",
            "function": {
                "name": "search_notes",
                "arguments": '{"query": "test"}',
            },
        },
    ]

    result = convert_to_anthropic_tool_invoke(tool_calls)

    assert len(result) == 2

    assert result[0]["id"] == "sessions_history_183"

    assert result[1]["id"] == "composio_NOTION_SEARCH"

    valid_tool_calls = [
        {
            "id": "toolu_01ABC-xyz_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "NYC"}',
            },
        }
    ]
    valid_result = convert_to_anthropic_tool_invoke(valid_tool_calls)
    assert valid_result[0]["id"] == "toolu_01ABC-xyz_123"


def test_convert_to_anthropic_tool_invoke_server_tool():
    """
    Test that a server tool call (srvtoolu_) with no stored result is replayed
    as a regular tool_use block.

    A server_tool_use block is only valid when paired with its result block, so
    an unpaired one must degrade to tool_use for Anthropic to accept the replay.
    A paired call still becomes server_tool_use, covered by
    test_convert_to_anthropic_tool_invoke_with_web_search_results.

    Context: https://github.com/BerriAI/litellm/issues/17737 (original
    server_tool_use reconstruction) and LIT-6622 / PR #39144 (unpaired calls
    degrade instead of 400ing at Anthropic).
    """
    tool_calls = [
        {
            "id": "srvtoolu_01ABC123",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": '{"query": "elephant weight"}',
            },
        }
    ]

    result = convert_to_anthropic_tool_invoke(tool_calls)

    assert len(result) == 1
    assert result[0]["type"] == "tool_use"
    assert result[0]["id"] == "srvtoolu_01ABC123"
    assert result[0]["name"] == "web_search"
    assert result[0]["input"] == {"query": "elephant weight"}


def test_convert_to_anthropic_tool_invoke_with_web_search_results():
    """
    Test that web_search_tool_result is included after server_tool_use.

    Fixes: https://github.com/BerriAI/litellm/issues/17737
    """
    tool_calls = [
        {
            "id": "srvtoolu_01ABC123",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": '{"query": "elephant weight"}',
            },
        }
    ]

    web_search_results = [
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_01ABC123",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://example.com",
                    "title": "Elephant Facts",
                    "snippet": "Elephants weigh 5000 kg",
                }
            ],
        }
    ]

    result = convert_to_anthropic_tool_invoke(tool_calls, web_search_results=web_search_results)

    assert len(result) == 2

    assert result[0]["type"] == "server_tool_use"
    assert result[0]["id"] == "srvtoolu_01ABC123"

    assert result[1]["type"] == "web_search_tool_result"
    assert result[1]["tool_use_id"] == "srvtoolu_01ABC123"


def test_convert_to_anthropic_tool_invoke_mixed_tools():
    """
    Test that mixed server and regular tools are reconstructed correctly.

    Fixes: https://github.com/BerriAI/litellm/issues/17737
    """
    tool_calls = [
        {
            "id": "srvtoolu_01ABC123",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": '{"query": "elephant weight"}',
            },
        },
        {
            "id": "toolu_01XYZ789",
            "type": "function",
            "function": {"name": "add_numbers", "arguments": '{"a": 5000, "b": 100}'},
        },
    ]

    web_search_results = [
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_01ABC123",
            "content": [{"url": "https://example.com", "title": "Test"}],
        }
    ]

    result = convert_to_anthropic_tool_invoke(tool_calls, web_search_results=web_search_results)

    assert len(result) == 3

    assert result[0]["type"] == "server_tool_use"
    assert result[0]["id"] == "srvtoolu_01ABC123"

    assert result[1]["type"] == "web_search_tool_result"

    assert result[2]["type"] == "tool_use"
    assert result[2]["id"] == "toolu_01XYZ789"


def test_anthropic_messages_pt_with_server_tool_use():
    """
    Test that anthropic_messages_pt correctly reconstructs server_tool_use from provider_specific_fields.

    Fixes: https://github.com/BerriAI/litellm/issues/17737
    """
    messages = [
        {"role": "user", "content": "Search for elephant weight and add 100"},
        {
            "role": "assistant",
            "content": "Let me search for that.",
            "tool_calls": [
                {
                    "id": "srvtoolu_01ABC123",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query": "elephant weight"}',
                    },
                },
                {
                    "id": "toolu_01XYZ789",
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "arguments": '{"a": 5000, "b": 100}',
                    },
                },
            ],
            "provider_specific_fields": {
                "web_search_results": [
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_01ABC123",
                        "content": [
                            {
                                "url": "https://example.com",
                                "title": "Test",
                                "snippet": "5000 kg",
                            }
                        ],
                    }
                ]
            },
        },
        {"role": "tool", "tool_call_id": "toolu_01XYZ789", "content": "5100"},
    ]

    result = anthropic_messages_pt(messages, model="claude-sonnet-4-5", llm_provider="anthropic")

    assistant_msg = next(m for m in result if m["role"] == "assistant")
    content = assistant_msg["content"]

    types = [c.get("type") for c in content]
    assert "text" in types
    assert "server_tool_use" in types
    assert "web_search_tool_result" in types
    assert "tool_use" in types

    server_tool = next(c for c in content if c.get("type") == "server_tool_use")
    assert server_tool["id"] == "srvtoolu_01ABC123"

    server_idx = types.index("server_tool_use")
    web_result_idx = types.index("web_search_tool_result")
    assert web_result_idx == server_idx + 1

    tool_use = next(c for c in content if c.get("type") == "tool_use")
    assert tool_use["id"] == "toolu_01XYZ789"


def test_convert_to_anthropic_tool_invoke_with_tool_results():
    """
    Test that non-web-search *_tool_result blocks (e.g. bash_code_execution_tool_result)
    stored in provider_specific_fields["tool_results"] are paired with their server_tool_use
    block when reconstructing assistant history.

    Regression for: server tool result blocks dropped on multi-turn replay
    (bash_code_execution_tool_result, text_editor_code_execution_tool_result, etc.)
    """
    tool_calls = [
        {
            "id": "srvtoolu_01BASH",
            "type": "function",
            "function": {
                "name": "bash_code_execution",
                "arguments": '{"command": "python3 -c \\"print(2)\\""}',
            },
        }
    ]

    tool_results = [
        {
            "type": "bash_code_execution_tool_result",
            "tool_use_id": "srvtoolu_01BASH",
            "content": {
                "type": "bash_code_execution_result",
                "stdout": "2\n",
                "stderr": "",
                "return_code": 0,
                "content": [],
            },
        }
    ]

    result = convert_to_anthropic_tool_invoke(tool_calls, tool_results=tool_results)

    assert len(result) == 2

    assert result[0]["type"] == "server_tool_use"
    assert result[0]["id"] == "srvtoolu_01BASH"
    assert result[0]["name"] == "bash_code_execution"

    assert result[1]["type"] == "bash_code_execution_tool_result"
    assert result[1]["tool_use_id"] == "srvtoolu_01BASH"


def test_anthropic_messages_pt_raw_bash_tool_result_passthrough():
    """
    Test that raw assistant content lists containing bash_code_execution_tool_result
    blocks are passed through intact to Anthropic.

    Regression: the raw-block passthrough only handled tool_search_tool_result;
    bash_code_execution_tool_result and other *_tool_result types were silently dropped.
    """
    messages = [
        {"role": "user", "content": "What is 1+1?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_01BASH",
                    "name": "bash_code_execution",
                    "input": {"command": 'python3 -c "print(1+1)"'},
                },
                {
                    "type": "bash_code_execution_tool_result",
                    "tool_use_id": "srvtoolu_01BASH",
                    "content": {
                        "type": "bash_code_execution_result",
                        "stdout": "2\n",
                        "stderr": "",
                        "return_code": 0,
                        "content": [],
                    },
                },
                {"type": "text", "text": "The answer is 2."},
            ],
        },
        {"role": "user", "content": "Thanks!"},
    ]

    result = anthropic_messages_pt(messages, model="claude-sonnet-4-5", llm_provider="anthropic")

    assistant_msg = next(m for m in result if m["role"] == "assistant")
    content = assistant_msg["content"]
    types = [c.get("type") for c in content]

    assert "server_tool_use" in types, "server_tool_use block must be preserved"
    assert "bash_code_execution_tool_result" in types, "bash_code_execution_tool_result block must not be dropped"
    assert "text" in types

    srv_idx = types.index("server_tool_use")
    result_idx = types.index("bash_code_execution_tool_result")
    assert result_idx == srv_idx + 1

    bash_result = next(c for c in content if c.get("type") == "bash_code_execution_tool_result")
    assert bash_result["tool_use_id"] == "srvtoolu_01BASH"


def test_anthropic_messages_pt_with_bash_tool_result_in_provider_specific_fields():
    """
    Test that anthropic_messages_pt correctly reconstructs bash_code_execution_tool_result
    from provider_specific_fields["tool_results"] when replaying LiteLLM response objects.

    Regression: only web_search_results were read from provider_specific_fields;
    tool_results (bash_code_execution_tool_result, etc.) were silently lost.
    """
    messages = [
        {"role": "user", "content": "What is 1+1?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "srvtoolu_01BASH",
                    "type": "function",
                    "function": {
                        "name": "bash_code_execution",
                        "arguments": '{"command": "python3 -c \\"print(1+1)\\""}',
                    },
                }
            ],
            "provider_specific_fields": {
                "tool_results": [
                    {
                        "type": "bash_code_execution_tool_result",
                        "tool_use_id": "srvtoolu_01BASH",
                        "content": {
                            "type": "bash_code_execution_result",
                            "stdout": "2\n",
                            "stderr": "",
                            "return_code": 0,
                            "content": [],
                        },
                    }
                ]
            },
        },
        {"role": "user", "content": "Thanks!"},
    ]

    result = anthropic_messages_pt(messages, model="claude-sonnet-4-5", llm_provider="anthropic")

    assistant_msg = next(m for m in result if m["role"] == "assistant")
    content = assistant_msg["content"]
    types = [c.get("type") for c in content]

    assert "server_tool_use" in types, "server_tool_use block must be reconstructed"
    assert "bash_code_execution_tool_result" in types, (
        "bash_code_execution_tool_result must be paired from provider_specific_fields['tool_results']"
    )

    srv_idx = types.index("server_tool_use")
    result_idx = types.index("bash_code_execution_tool_result")
    assert result_idx == srv_idx + 1

    srv = next(c for c in content if c.get("type") == "server_tool_use")
    assert srv["id"] == "srvtoolu_01BASH"
    bash_result = next(c for c in content if c.get("type") == "bash_code_execution_tool_result")
    assert bash_result["tool_use_id"] == "srvtoolu_01BASH"


def test_parse_tool_call_arguments_valid_json():
    """Test that valid JSON is parsed correctly."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        parse_tool_call_arguments,
    )

    result = parse_tool_call_arguments('{"city": "Paris", "units": "celsius"}')
    assert result == {"city": "Paris", "units": "celsius"}


def test_parse_tool_call_arguments_empty_input():
    """Test that None/empty input returns empty dict."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        parse_tool_call_arguments,
    )

    assert parse_tool_call_arguments(None) == {}
    assert parse_tool_call_arguments("") == {}


def test_parse_tool_call_arguments_malformed_json():
    """Test that malformed JSON raises ValueError with context."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        parse_tool_call_arguments,
    )

    with pytest.raises(ValueError, match="Failed to parse tool call arguments for tool 'load_skill") as exc_info:
        parse_tool_call_arguments(
            '{"skill_name": "pptx',
            tool_name="load_skill",
            context="Anthropic tool invoke",
        )

    error_msg = str(exc_info.value)
    assert "load_skill" in error_msg
    assert "Anthropic tool invoke" in error_msg
    assert '{"skill_name": "pptx' in error_msg
    assert "Unterminated string" in error_msg


def test_convert_to_anthropic_tool_invoke_malformed_json():
    """
    Test that convert_to_anthropic_tool_invoke raises ValueError with context
    when tool arguments contain malformed JSON.

    Fixes: https://github.com/BerriAI/litellm/issues/18920
    """
    tool_calls = [
        {
            "id": "toolu_01_invalid",
            "type": "function",
            "function": {
                "name": "bad_tool",
                "arguments": '{"truncated',
            },
        }
    ]

    with pytest.raises(ValueError, match="Failed to parse tool call arguments for tool 'bad_tool") as exc_info:
        convert_to_anthropic_tool_invoke(tool_calls)

    error_msg = str(exc_info.value)
    assert "bad_tool" in error_msg
    assert '{"truncated' in error_msg


def test_attempt_json_repair_missing_closing_brace():
    """Repair JSON truncated with a missing closing brace (issue #22312)."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        _attempt_json_repair,
    )

    truncated = '{"command": ["bash","-lc","find /x/repos -name \'messages.py\' -type f"]'
    result = _attempt_json_repair(truncated)
    assert result is not None
    assert result["command"] == [
        "bash",
        "-lc",
        "find /x/repos -name 'messages.py' -type f",
    ]


def test_attempt_json_repair_missing_bracket_and_brace():
    """Repair JSON truncated with both missing ] and }."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        _attempt_json_repair,
    )

    truncated = '{"items": [1, 2, 3'
    result = _attempt_json_repair(truncated)
    assert result is not None
    assert result["items"] == [1, 2, 3]


def test_attempt_json_repair_trailing_comma():
    """Repair JSON with a trailing comma before missing close."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        _attempt_json_repair,
    )

    truncated = '{"a": 1, "b": 2,'
    result = _attempt_json_repair(truncated)
    assert result is not None
    assert result == {"a": 1, "b": 2}


def test_attempt_json_repair_returns_none_for_unterminated_string():
    """Cannot repair an unterminated string — returns None."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        _attempt_json_repair,
    )

    assert _attempt_json_repair('{"key": "incomplete value') is None


def test_attempt_json_repair_returns_none_for_valid_json():
    """Valid JSON has no unmatched brackets — returns None (no repair needed)."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        _attempt_json_repair,
    )

    assert _attempt_json_repair('{"key": "value"}') is None


def test_attempt_json_repair_returns_none_for_empty():
    """Empty / whitespace input returns None."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        _attempt_json_repair,
    )

    assert _attempt_json_repair("") is None
    assert _attempt_json_repair("   ") is None


def test_attempt_json_repair_interleaved_nesting():
    """Repair JSON with interleaved {} and [] nesting."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        _attempt_json_repair,
    )

    truncated = '{"a": [{"b": 2'
    result = _attempt_json_repair(truncated)
    assert result is not None
    assert result == {"a": [{"b": 2}]}


def test_attempt_json_repair_deeply_nested():
    """Repair deeply nested truncated JSON."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        _attempt_json_repair,
    )

    truncated = '{"x": {"y": [1, {"z": [2, 3'
    result = _attempt_json_repair(truncated)
    assert result is not None
    assert result == {"x": {"y": [1, {"z": [2, 3]}]}}


def test_parse_tool_call_arguments_whitespace_only():
    """Whitespace-only input returns empty dict."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        parse_tool_call_arguments,
    )

    assert parse_tool_call_arguments("   ") == {}
    assert parse_tool_call_arguments("\n") == {}


def test_parse_tool_call_arguments_non_object_json():
    """Non-object JSON (list, string, number) is returned as-is (no wrapping)."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        parse_tool_call_arguments,
    )

    result = parse_tool_call_arguments("[1, 2, 3]")
    assert result == [1, 2, 3]


def test_parse_tool_call_arguments_repairs_truncated_json():
    """parse_tool_call_arguments should repair truncated JSON instead of raising."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        parse_tool_call_arguments,
    )

    truncated = '{"command": ["bash","-lc","find /x -type f"]'
    result = parse_tool_call_arguments(truncated, tool_name="shell", context="Anthropic tool invoke")
    assert result == {"command": ["bash", "-lc", "find /x -type f"]}


def test_parse_tool_call_arguments_still_raises_for_unrepairable():
    """parse_tool_call_arguments raises ValueError when repair also fails."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        parse_tool_call_arguments,
    )

    with pytest.raises(ValueError, match="Failed to parse tool call arguments for tool 'test_tool") as exc_info:
        parse_tool_call_arguments(
            '{"key": "unterminated',
            tool_name="test_tool",
            context="test context",
        )

    error_msg = str(exc_info.value)
    assert "test_tool" in error_msg
    assert "test context" in error_msg


def test_anthropic_messages_pt_interleave_thinking_with_server_tool_calls():
    """
    Test that thinking blocks are interleaved with server tool calls (web search)
    instead of being prepended all at once.

    When Anthropic returns a response with extended thinking + multiple web searches,
    the content blocks are interleaved:
    [thinking_1, server_tool_use_1, result_1, thinking_2, server_tool_use_2, result_2]

    On round-trip through OpenAI format, thinking_blocks and tool_calls are separate
    fields. anthropic_messages_pt must reconstruct the interleaved order, otherwise
    Anthropic rejects the request because thinking block signatures are position-dependent.

    Fixes: https://github.com/BerriAI/litellm/issues/23047
    """
    messages = [
        {"role": "user", "content": "Search for news about fast.ai and answer.ai"},
        {
            "role": "assistant",
            "content": "Here is what I found.",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "I need to search for fast.ai news.",
                    "signature": "sig_thinking_1",
                },
                {
                    "type": "thinking",
                    "thinking": "Now I should also search for answer.ai.",
                    "signature": "sig_thinking_2",
                },
            ],
            "tool_calls": [
                {
                    "id": "srvtoolu_01SEARCH1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query": "fast.ai news"}',
                    },
                },
                {
                    "id": "srvtoolu_01SEARCH2",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query": "answer.ai news"}',
                    },
                },
            ],
            "provider_specific_fields": {
                "web_search_results": [
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_01SEARCH1",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://fast.ai",
                                "title": "fast.ai",
                                "snippet": "fast.ai news",
                            }
                        ],
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_01SEARCH2",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://answer.ai",
                                "title": "answer.ai",
                                "snippet": "answer.ai news",
                            }
                        ],
                    },
                ]
            },
        },
        {"role": "user", "content": "Now search for news about solveit"},
    ]

    result = anthropic_messages_pt(messages, model="claude-sonnet-4-5", llm_provider="anthropic")

    assistant_msg = next(m for m in result if m["role"] == "assistant")
    content = assistant_msg["content"]

    types = [c.get("type") for c in content]

    assert types == [
        "thinking",
        "server_tool_use",
        "web_search_tool_result",
        "thinking",
        "server_tool_use",
        "web_search_tool_result",
        "text",
    ], f"Expected interleaved order but got: {types}"

    thinking_1 = content[0]
    assert thinking_1["thinking"] == "I need to search for fast.ai news."
    assert thinking_1["signature"] == "sig_thinking_1"

    thinking_2 = content[3]
    assert thinking_2["thinking"] == "Now I should also search for answer.ai."
    assert thinking_2["signature"] == "sig_thinking_2"

    assert content[1]["id"] == "srvtoolu_01SEARCH1"
    assert content[4]["id"] == "srvtoolu_01SEARCH2"

    assert content[2]["tool_use_id"] == "srvtoolu_01SEARCH1"
    assert content[5]["tool_use_id"] == "srvtoolu_01SEARCH2"

    assert content[6]["text"] == "Here is what I found."


def test_anthropic_messages_pt_thinking_blocks_no_server_tools_unchanged():
    """
    Test that the existing behavior is preserved when thinking blocks exist
    but there are no server tool calls (only regular tool_use).

    Thinking blocks should still be prepended first in this case.
    """
    messages = [
        {"role": "user", "content": "What is the weather?"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "I should check the weather.",
                    "signature": "sig_1",
                },
            ],
            "tool_calls": [
                {
                    "id": "toolu_01REG",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "SF"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_01REG",
            "content": "72F and sunny",
        },
    ]

    result = anthropic_messages_pt(messages, model="claude-sonnet-4-5", llm_provider="anthropic")

    assistant_msg = next(m for m in result if m["role"] == "assistant")
    content = assistant_msg["content"]
    types = [c.get("type") for c in content]

    assert types == [
        "thinking",
        "text",
        "tool_use",
    ], f"Expected sequential order but got: {types}"


def test_anthropic_messages_pt_interleave_more_thinking_than_tool_groups():
    """
    Test interleaving when there are more thinking blocks than server tool groups.
    Extra thinking blocks should appear before the text block.
    """
    messages = [
        {"role": "user", "content": "Search for something"},
        {
            "role": "assistant",
            "content": "Found it.",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "First thought",
                    "signature": "sig_1",
                },
                {
                    "type": "thinking",
                    "thinking": "Second thought",
                    "signature": "sig_2",
                },
                {
                    "type": "thinking",
                    "thinking": "Third thought after search",
                    "signature": "sig_3",
                },
            ],
            "tool_calls": [
                {
                    "id": "srvtoolu_01ONLY",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query": "something"}',
                    },
                },
            ],
            "provider_specific_fields": {
                "web_search_results": [
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_01ONLY",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://example.com",
                                "title": "Test",
                                "snippet": "result",
                            }
                        ],
                    },
                ]
            },
        },
    ]

    result = anthropic_messages_pt(messages, model="claude-sonnet-4-5", llm_provider="anthropic")

    assistant_msg = next(m for m in result if m["role"] == "assistant")
    content = assistant_msg["content"]
    types = [c.get("type") for c in content]

    assert types == [
        "thinking",
        "server_tool_use",
        "web_search_tool_result",
        "thinking",
        "thinking",
        "text",
    ], f"Expected order but got: {types}"


def test_anthropic_messages_pt_list_content_with_thinking_preserves_order():
    """
    Test that when assistant content is already a list containing interleaved
    thinking blocks and server tool blocks, the thinking_blocks from
    provider_specific_fields are NOT duplicated/prepended.

    This covers the gap identified by Greptile where list-content messages
    bypass INTERLEAVED MODE and fall into SEQUENTIAL MODE, which previously
    would prepend all thinking_blocks again, causing duplication and
    breaking Anthropic's position-dependent signature verification.

    Fixes: https://github.com/BerriAI/litellm/issues/23047
    """
    messages = [
        {"role": "user", "content": "Search for AI news"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me search for AI news.",
                    "signature": "sig_1",
                },
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_01SEARCH1",
                    "name": "web_search",
                    "input": {"query": "AI news"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_01SEARCH1",
                    "content": [
                        {
                            "type": "web_search_result",
                            "url": "https://example.com",
                            "title": "AI News",
                            "snippet": "Latest AI news",
                        }
                    ],
                },
                {
                    "type": "thinking",
                    "thinking": "Now let me summarize.",
                    "signature": "sig_2",
                },
                {
                    "type": "text",
                    "text": "Here is the AI news summary.",
                },
            ],
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Let me search for AI news.",
                    "signature": "sig_1",
                },
                {
                    "type": "thinking",
                    "thinking": "Now let me summarize.",
                    "signature": "sig_2",
                },
            ],
        },
        {"role": "user", "content": "Tell me more"},
    ]

    result = anthropic_messages_pt(messages, model="claude-sonnet-4-5", llm_provider="anthropic")

    assistant_msg = next(m for m in result if m["role"] == "assistant")
    content = assistant_msg["content"]
    types = [c.get("type") for c in content]

    assert types == [
        "thinking",
        "server_tool_use",
        "web_search_tool_result",
        "thinking",
        "text",
    ], f"Expected preserved list order without duplicate thinking blocks, but got: {types}"

    thinking_count = sum(1 for t in types if t == "thinking")
    assert thinking_count == 2, f"Expected 2 thinking blocks, got {thinking_count} (duplication detected)"

    assert content[0]["signature"] == "sig_1"
    assert content[3]["signature"] == "sig_2"


def test_get_tool_calls_from_response_chat_completions():
    response = MagicMock()
    response.output = None
    response.content = None
    tool_call = MagicMock()
    tool_call.id = "call_abc"
    tool_call.function.name = "my_tool"
    tool_call.function.arguments = '{"x": 1}'
    response.choices = [MagicMock(message=MagicMock(tool_calls=[tool_call]))]

    result = get_tool_calls_from_response(response)

    assert result == [{"id": "call_abc", "name": "my_tool", "arguments": {"x": 1}}]


def test_get_tool_calls_from_response_responses_api():
    response = MagicMock()
    response.choices = None
    response.content = None
    response.output = [
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "my_tool",
            "arguments": '{"x": 2}',
        }
    ]

    result = get_tool_calls_from_response(response)

    assert result == [{"id": "call_1", "name": "my_tool", "arguments": {"x": 2}}]


def test_get_tool_calls_from_response_anthropic_messages():
    response = MagicMock()
    response.choices = None
    response.output = None
    response.content = [
        {"type": "tool_use", "id": "toolu_1", "name": "my_tool", "input": {"x": 3}},
    ]

    result = get_tool_calls_from_response(response)

    assert result == [{"id": "toolu_1", "name": "my_tool", "arguments": {"x": 3}}]


def test_get_tool_calls_from_response_anthropic_messages_plain_dict():

    response = {
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "my_tool", "input": {"x": 3}},
        ]
    }

    result = get_tool_calls_from_response(response)

    assert result == [{"id": "toolu_1", "name": "my_tool", "arguments": {"x": 3}}]


def test_get_tool_calls_from_response_no_tool_calls():
    response = MagicMock()
    response.choices = None
    response.output = None
    response.content = None

    assert get_tool_calls_from_response(response) == []


def test_has_tool_with_name_openai_function_shape():
    tools = [{"type": "function", "function": {"name": "my_tool"}}]
    assert has_tool_with_name(tools, "my_tool")
    assert not has_tool_with_name(tools, "other_tool")


def test_has_tool_with_name_anthropic_custom_shape():
    tools = [{"type": "custom", "name": "my_tool", "input_schema": {}}]
    assert has_tool_with_name(tools, "my_tool")
    assert not has_tool_with_name(tools, "other_tool")


def test_has_tool_with_name_anthropic_shape_without_type_field():

    tools = [{"name": "my_tool", "input_schema": {}}]
    assert has_tool_with_name(tools, "my_tool")
    assert not has_tool_with_name(tools, "other_tool")


def test_has_tool_with_name_not_a_list():
    assert not has_tool_with_name(None, "my_tool")
    assert not has_tool_with_name("not a list", "my_tool")


def test_completion_bedrock_invalid_role_exception(monkeypatch):
    """
    Test if litellm raises a BadRequestError for an invalid role on Bedrock
    """
    monkeypatch.setattr(litellm, "set_verbose", True)
    with pytest.raises(litellm.BadRequestError) as exc_info:
        litellm.completion(
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{"role": "very-bad-role", "content": "hello"}],
        )

    assert (
        str(exc_info.value)
        == "litellm.BadRequestError: Invalid Message passed in {'role': 'very-bad-role', 'content': 'hello'}"
    )
