import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_format_from_file_id,
    handle_any_messages_to_chat_completion_str_messages_conversion,
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
