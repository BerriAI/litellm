import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    add_system_prompt_to_messages,
    get_file_ids_from_messages,
    get_format_from_file_id,
    handle_any_messages_to_chat_completion_str_messages_conversion,
    split_concatenated_json_objects,
    update_messages_with_model_file_ids,
)


def test_get_format_from_file_id():
    unified_file_id = (
        "litellm_proxy:application/pdf;unified_id,cbbe3534-8bf8-4386-af00-f5f6b7e370bf"
    )

    format = get_format_from_file_id(unified_file_id)

    assert format == "application/pdf"


def test_update_messages_with_model_file_ids():
    file_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9wZGY7dW5pZmllZF9pZCxmYzdmMmVhNS0wZjUwLTQ5ZjYtODljMS03ZTZhNTRiMTIxMzg"
    model_id = "my_model_id"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "file",
                    "file": {
                        "file_id": file_id,
                    },
                },
            ],
        },
    ]

    model_file_id_mapping = {file_id: {"my_model_id": "provider_file_id"}}

    updated_messages = update_messages_with_model_file_ids(
        messages, model_id, model_file_id_mapping
    )

    assert updated_messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "file",
                    "file": {
                        "file_id": "provider_file_id",
                        "format": "application/pdf",
                    },
                },
            ],
        }
    ]


def test_handle_any_messages_to_chat_completion_str_messages_conversion_list():
    # Test with list of messages
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = handle_any_messages_to_chat_completion_str_messages_conversion(messages)
    assert len(result) == 2
    assert result[0] == messages[0]
    assert result[1] == messages[1]


def test_handle_any_messages_to_chat_completion_str_messages_conversion_list_infinite_loop():
    # Test that list handling doesn't cause infinite recursion
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    # This should complete without stack overflow
    result = handle_any_messages_to_chat_completion_str_messages_conversion(messages)
    assert len(result) == 2
    assert result[0] == messages[0]
    assert result[1] == messages[1]


def test_handle_any_messages_to_chat_completion_str_messages_conversion_dict():
    # Test with single dictionary message
    message = {"role": "user", "content": "Hello"}
    result = handle_any_messages_to_chat_completion_str_messages_conversion(message)
    assert len(result) == 1
    assert result[0]["input"] == json.dumps(message)


def test_handle_any_messages_to_chat_completion_str_messages_conversion_str():
    # Test with string message
    message = "Hello"
    result = handle_any_messages_to_chat_completion_str_messages_conversion(message)
    assert len(result) == 1
    assert result[0]["input"] == message


def test_handle_any_messages_to_chat_completion_str_messages_conversion_other():
    # Test with non-string/dict/list type
    message = 123
    result = handle_any_messages_to_chat_completion_str_messages_conversion(message)
    assert len(result) == 1
    assert result[0]["input"] == "123"


def test_handle_any_messages_to_chat_completion_str_messages_conversion_complex():
    # Test with complex nested structure
    message = {
        "role": "user",
        "content": {"text": "Hello", "metadata": {"timestamp": "2024-01-01"}},
    }
    result = handle_any_messages_to_chat_completion_str_messages_conversion(message)
    assert len(result) == 1
    assert result[0]["input"] == json.dumps(message)


def test_add_system_prompt_to_messages_prepend():
    """Adds system prompt at beginning when no system message exists."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = add_system_prompt_to_messages(messages, "You are a helpful assistant.")
    assert result == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


def test_add_system_prompt_to_messages_empty_prompt_unchanged():
    """Returns messages unchanged when system_prompt is empty."""
    messages = [{"role": "user", "content": "Hello"}]
    assert add_system_prompt_to_messages(messages, "") == messages
    assert add_system_prompt_to_messages(messages, None) == messages


def test_add_system_prompt_to_messages_merge_with_first_system():
    """Merges new prompt into first system message when merge_with_first_system=True."""
    messages = [
        {"role": "system", "content": "Existing system prompt."},
        {"role": "user", "content": "Hello"},
    ]
    result = add_system_prompt_to_messages(
        messages, "You are helpful.", merge_with_first_system=True
    )
    assert result == [
        {"role": "system", "content": "You are helpful.\n\nExisting system prompt."},
        {"role": "user", "content": "Hello"},
    ]


def test_add_system_prompt_to_messages_merge_with_first_system_adds_new_when_no_system():
    """When merge_with_first_system=True but no system message, adds new one at start."""
    messages = [{"role": "user", "content": "Hello"}]
    result = add_system_prompt_to_messages(
        messages, "You are helpful.", merge_with_first_system=True
    )
    assert result == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]


def test_add_system_prompt_to_messages_empty_list():
    """Adds system prompt to empty messages list."""
    result = add_system_prompt_to_messages([], "You are helpful.")
    assert result == [{"role": "system", "content": "You are helpful."}]


def test_convert_prefix_message_to_non_prefix_messages():
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        convert_prefix_message_to_non_prefix_messages,
    )

    messages = [
        {"role": "assistant", "content": "value", "prefix": True},
    ]
    result = convert_prefix_message_to_non_prefix_messages(messages)
    assert result == [
        {
            "role": "system",
            "content": "You are a helpful assistant. You are given a message and you need to respond to it. You are also given a generated content. You need to respond to the message in continuation of the generated content. Do not repeat the same content. Your response should be in continuation of this text: ",
        },
        {"role": "assistant", "content": "value"},
    ]


# ── split_concatenated_json_objects tests ──


def test_split_concatenated_json_single_object():
    """A single valid JSON object is returned as a one-element list."""
    result = split_concatenated_json_objects('{"location": "Boston"}')
    assert result == [{"location": "Boston"}]


def test_split_concatenated_json_multiple_objects():
    """
    Multiple JSON objects concatenated without separators are split correctly.
    This is the exact pattern from issue #20543 where Bedrock Claude Sonnet 4.5
    returns concatenated JSON in a single tool call arguments string.
    """
    raw = (
        '{"command": ["curl", "-i", "http://localhost:9009"]}'
        '{"command": ["curl", "-i", "http://localhost:9009/robots.txt"]}'
        '{"command": ["curl", "-i", "http://localhost:9009/sitemap.xml"]}'
    )
    result = split_concatenated_json_objects(raw)
    assert len(result) == 3
    assert result[0] == {"command": ["curl", "-i", "http://localhost:9009"]}
    assert result[1] == {"command": ["curl", "-i", "http://localhost:9009/robots.txt"]}
    assert result[2] == {"command": ["curl", "-i", "http://localhost:9009/sitemap.xml"]}


def test_split_concatenated_json_with_whitespace():
    """Objects separated by whitespace are handled correctly."""
    raw = '{"a": 1}  {"b": 2}\n{"c": 3}'
    result = split_concatenated_json_objects(raw)
    assert len(result) == 3
    assert result[0] == {"a": 1}
    assert result[1] == {"b": 2}
    assert result[2] == {"c": 3}


def test_split_concatenated_json_empty_string():
    """Empty or whitespace-only strings return an empty list."""
    assert split_concatenated_json_objects("") == []
    assert split_concatenated_json_objects("   ") == []


def test_split_concatenated_json_non_dict_value():
    """Non-dict JSON values (e.g. arrays, strings) are replaced with {}."""
    result = split_concatenated_json_objects("[1, 2, 3]")
    assert result == [{}]


def test_split_concatenated_json_invalid_raises():
    """Completely invalid JSON raises JSONDecodeError."""
    with pytest.raises(json.JSONDecodeError):
        split_concatenated_json_objects("not json at all")


# ---------------------------------------------------------------------------
# Regression tests for non-OpenAI file content blocks.
#
# `type: "file"` is a public content-block discriminator. Several producers
# (LangChain v1, provider-native shapes, custom user code) emit blocks with
# `type: "file"` but without the OpenAI Chat Completions `file` sub-dict.
# The discovery helpers below are used unconditionally inside
# `AnthropicConfig.validate_environment`, so any crash there surfaces as a
# `500 InternalServerError` before the request is even dispatched.
# ---------------------------------------------------------------------------


def test_get_file_ids_from_messages_skips_langchain_v1_file_block():
    """A LangChain v1 standardized file block must not crash file-id discovery."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "summarise this PDF"},
                # LangChain v1 shape produced by `_normalize_messages`.
                # No `file` sub-dict: the discriminator is `type: "file"` but
                # the payload lives on `base64`/`mime_type` siblings.
                {
                    "type": "file",
                    "id": "lc_1",
                    "base64": "JVBERi0xLjQK",
                    "mime_type": "application/pdf",
                    "extras": {"file_format": "application/pdf"},
                },
            ],
        }
    ]

    assert get_file_ids_from_messages(messages) == []


def test_get_file_ids_from_messages_still_extracts_from_openai_shape():
    """Well-formed OpenAI file blocks still yield their file_id."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "file", "file": {"file_id": "file-abc"}},
            ],
        }
    ]

    assert get_file_ids_from_messages(messages) == ["file-abc"]


def test_get_file_ids_from_messages_mixed_shapes():
    """Mixed OpenAI and non-OpenAI file blocks: extract from the former,
    ignore the latter."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "file", "file": {"file_id": "file-keep"}},
                {
                    "type": "file",
                    "id": "lc_2",
                    "base64": "AAA",
                    "mime_type": "application/pdf",
                },
            ],
        }
    ]

    assert get_file_ids_from_messages(messages) == ["file-keep"]


def test_get_file_ids_from_messages_file_field_not_dict():
    """`file` set to a non-dict value (e.g. stringified payload) must not crash."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "file", "file": "unexpectedly-a-string"},
            ],
        }
    ]

    assert get_file_ids_from_messages(messages) == []


def test_update_messages_with_model_file_ids_skips_non_openai_file_blocks():
    """`update_messages_with_model_file_ids` is also called on user content
    before provider dispatch. It must tolerate non-OpenAI file blocks the same
    way."""
    langchain_v1_block = {
        "type": "file",
        "id": "lc_3",
        "base64": "AAA",
        "mime_type": "application/pdf",
    }
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                langchain_v1_block,
            ],
        }
    ]

    updated = update_messages_with_model_file_ids(messages, "model-1", {})

    # Messages pass through unchanged when there is no `file` sub-dict to remap.
    assert updated == messages
