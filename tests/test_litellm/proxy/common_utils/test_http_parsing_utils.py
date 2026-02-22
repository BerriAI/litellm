import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path


import litellm
from litellm.proxy._types import ProxyException
from litellm.proxy.common_utils.http_parsing_utils import (
    _read_request_body,
    _safe_get_request_headers,
    _safe_get_request_parsed_body,
    _safe_get_request_query_params,
    _safe_set_request_parsed_body,
    get_form_data,
    get_request_body,
    get_tags_from_request_body,
    populate_request_with_path_params,
)


@pytest.mark.asyncio
async def test_request_body_caching():
    """
    Test that the request body is cached after the first read and subsequent
    calls use the cached version instead of parsing again.
    """
    # Create a mock request with a JSON body
    mock_request = MagicMock()
    test_data = {"key": "value"}
    # Use AsyncMock for the body method
    mock_request.body = AsyncMock(return_value=orjson.dumps(test_data))
    mock_request.headers = {"content-type": "application/json"}
    mock_request.scope = {}

    # First call should parse the body
    result1 = await _read_request_body(mock_request)
    assert result1 == test_data
    assert "parsed_body" in mock_request.scope
    assert mock_request.scope["parsed_body"] == (("key",), {"key": "value"})

    # Verify the body was read once
    mock_request.body.assert_called_once()

    # Reset the mock to track the second call
    mock_request.body.reset_mock()

    # Second call should use the cached body
    result2 = await _read_request_body(mock_request)
    assert result2 == {"key": "value"}

    # Verify the body was not read again
    mock_request.body.assert_not_called()


@pytest.mark.asyncio
async def test_form_data_parsing():
    """
    Test that form data is correctly parsed from the request.
    """
    # Create a mock request with form data
    mock_request = MagicMock()
    test_data = {"name": "test_user", "message": "hello world"}

    # Mock the form method to return the test data as an awaitable
    mock_request.form = AsyncMock(return_value=test_data)
    mock_request.headers = {"content-type": "application/x-www-form-urlencoded"}
    mock_request.scope = {}

    # Parse the form data
    result = await _read_request_body(mock_request)

    # Verify the form data was correctly parsed
    assert result == test_data
    assert "parsed_body" in mock_request.scope
    assert mock_request.scope["parsed_body"] == (
        ("name", "message"),
        {"name": "test_user", "message": "hello world"},
    )

    # Verify form() was called
    mock_request.form.assert_called_once()

    # The body method should not be called for form data
    assert not hasattr(mock_request, "body") or not mock_request.body.called


@pytest.mark.asyncio
async def test_form_data_with_json_metadata():
    """
    Test that form data with a JSON-encoded metadata field is correctly parsed.
    
    When form data includes a 'metadata' field, it comes as a JSON string that needs
    to be parsed into a Python dictionary (lines 42-43 of http_parsing_utils.py).
    """
    # Create a mock request with form data containing JSON metadata
    mock_request = MagicMock()
    
    # Metadata is sent as a JSON string in form data
    metadata_json_string = json.dumps({
        "user_id": "12345",
        "request_type": "audio_transcription",
        "tags": ["urgent", "production"],
        "custom_field": {"nested": "value"}
    })
    
    test_data = {
        "model": "whisper-1",
        "file": "audio.mp3",
        "metadata": metadata_json_string  # This is a JSON string, not a dict
    }

    # Mock the form method to return the test data as an awaitable
    mock_request.form = AsyncMock(return_value=test_data)
    mock_request.headers = {"content-type": "multipart/form-data"}
    mock_request.scope = {}

    # Parse the form data
    result = await _read_request_body(mock_request)

    # Verify the metadata was parsed from JSON string to dict
    assert "metadata" in result
    assert isinstance(result["metadata"], dict)
    assert result["metadata"]["user_id"] == "12345"
    assert result["metadata"]["request_type"] == "audio_transcription"
    assert result["metadata"]["tags"] == ["urgent", "production"]
    assert result["metadata"]["custom_field"] == {"nested": "value"}
    
    # Verify other fields remain unchanged
    assert result["model"] == "whisper-1"
    assert result["file"] == "audio.mp3"
    
    # Verify form() was called
    mock_request.form.assert_called_once()


@pytest.mark.asyncio
async def test_form_data_with_invalid_json_metadata():
    """
    Test that form data with invalid JSON in metadata field raises an exception.
    
    This tests error handling when the metadata field contains malformed JSON.
    """
    # Create a mock request with form data containing invalid JSON metadata
    mock_request = MagicMock()
    
    test_data = {
        "model": "whisper-1",
        "file": "audio.mp3",
        "metadata": '{"invalid": json}'  # Invalid JSON - unquoted value
    }

    # Mock the form method to return the test data
    mock_request.form = AsyncMock(return_value=test_data)
    mock_request.headers = {"content-type": "multipart/form-data"}
    mock_request.scope = {}

    # Should raise JSONDecodeError when trying to parse invalid JSON metadata
    with pytest.raises(json.JSONDecodeError):
        await _read_request_body(mock_request)


@pytest.mark.asyncio
async def test_form_data_without_metadata():
    """
    Test that form data without metadata field works correctly.
    
    Ensures the metadata parsing logic doesn't break when metadata is absent.
    """
    # Create a mock request with form data without metadata
    mock_request = MagicMock()
    
    test_data = {
        "model": "whisper-1",
        "file": "audio.mp3",
        "language": "en"
    }

    # Mock the form method to return the test data
    mock_request.form = AsyncMock(return_value=test_data)
    mock_request.headers = {"content-type": "application/x-www-form-urlencoded"}
    mock_request.scope = {}

    # Parse the form data
    result = await _read_request_body(mock_request)

    # Verify all fields are preserved as-is
    assert result == test_data
    assert "metadata" not in result
    assert result["model"] == "whisper-1"
    assert result["file"] == "audio.mp3"
    assert result["language"] == "en"


@pytest.mark.asyncio
async def test_form_data_with_empty_metadata():
    """
    Test that form data with empty JSON object in metadata field is parsed correctly.
    """
    # Create a mock request with form data containing empty metadata
    mock_request = MagicMock()
    
    test_data = {
        "model": "whisper-1",
        "file": "audio.mp3",
        "metadata": "{}"  # Empty JSON object as string
    }

    # Mock the form method to return the test data
    mock_request.form = AsyncMock(return_value=test_data)
    mock_request.headers = {"content-type": "multipart/form-data"}
    mock_request.scope = {}

    # Parse the form data
    result = await _read_request_body(mock_request)

    # Verify the metadata was parsed to an empty dict
    assert "metadata" in result
    assert isinstance(result["metadata"], dict)
    assert result["metadata"] == {}
    assert result["model"] == "whisper-1"


@pytest.mark.asyncio
async def test_form_data_with_dict_metadata():
    """
    Test that form data with metadata already as a dict is not parsed again.
    
    This handles edge cases where metadata might already be a dictionary
    (shouldn't happen in normal form data, but defensive coding).
    """
    # Create a mock request with form data where metadata is already a dict
    mock_request = MagicMock()
    
    metadata_dict = {
        "user_id": "12345",
        "tags": ["test"]
    }
    
    test_data = {
        "model": "whisper-1",
        "file": "audio.mp3",
        "metadata": metadata_dict  # Already a dict, not a string
    }

    # Mock the form method to return the test data
    mock_request.form = AsyncMock(return_value=test_data)
    mock_request.headers = {"content-type": "multipart/form-data"}
    mock_request.scope = {}

    # Parse the form data
    result = await _read_request_body(mock_request)

    # Verify the metadata remains as a dict and is not parsed
    assert "metadata" in result
    assert isinstance(result["metadata"], dict)
    assert result["metadata"] == metadata_dict
    assert result["metadata"]["user_id"] == "12345"
    assert result["model"] == "whisper-1"


@pytest.mark.asyncio
async def test_form_data_with_none_metadata():
    """
    Test that form data with None metadata value is handled gracefully.
    """
    # Create a mock request with form data where metadata is None
    mock_request = MagicMock()
    
    test_data = {
        "model": "whisper-1",
        "file": "audio.mp3",
        "metadata": None  # None value
    }

    # Mock the form method to return the test data
    mock_request.form = AsyncMock(return_value=test_data)
    mock_request.headers = {"content-type": "multipart/form-data"}
    mock_request.scope = {}

    # Parse the form data
    result = await _read_request_body(mock_request)

    # Verify the metadata remains None (not parsed)
    assert "metadata" in result
    assert result["metadata"] is None
    assert result["model"] == "whisper-1"


@pytest.mark.asyncio
async def test_empty_request_body():
    """
    Test handling of empty request bodies.
    """
    # Create a mock request with an empty body
    mock_request = MagicMock()
    mock_request.body = AsyncMock(return_value=b"")  # Empty bytes as an awaitable
    mock_request.headers = {"content-type": "application/json"}
    mock_request.scope = {}

    # Parse the empty body
    result = await _read_request_body(mock_request)

    # Verify an empty dict is returned
    assert result == {}
    assert "parsed_body" in mock_request.scope
    assert mock_request.scope["parsed_body"] == ((), {})

    # Verify the body was read
    mock_request.body.assert_called_once()


@pytest.mark.asyncio
async def test_circular_reference_handling():
    """
    Test that cached request body isn't modified when the returned result is modified.
    Demonstrates the mutable dictionary reference issue.
    """
    # Create a mock request with initial data
    mock_request = MagicMock()
    initial_body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    mock_request.body = AsyncMock(return_value=orjson.dumps(initial_body))
    mock_request.headers = {"content-type": "application/json"}
    mock_request.scope = {}

    # First parse
    result = await _read_request_body(mock_request)

    # Verify initial parse
    assert result["model"] == "gpt-4"
    assert result["messages"] == [{"role": "user", "content": "Hello"}]

    # Modify the result by adding proxy_server_request
    result["proxy_server_request"] = {
        "url": "http://0.0.0.0:4000/v1/chat/completions",
        "method": "POST",
        "headers": {"content-type": "application/json"},
        "body": result,  # Creates circular reference
    }

    # Second parse using the same request - will use the modified cached value
    result2 = await _read_request_body(mock_request)
    assert (
        "proxy_server_request" not in result2
    )  # This will pass, showing the cache pollution


@pytest.mark.asyncio
async def test_json_parsing_error_handling():
    """
    Test that JSON parsing errors are properly handled and raise ProxyException
    with appropriate error messages.
    """
    # Test case 1: Trailing comma error
    mock_request = MagicMock()
    invalid_json_with_trailing_comma = b'''{
        "model": "gpt-4o",
        "tools": [
            {
                "type": "mcp",
                "server_label": "litellm",
                "headers": {
                    "x-litellm-api-key": "Bearer sk-1234",
                }
            }
        ],
        "input": "Run available tools"
    }'''
    
    mock_request.body = AsyncMock(return_value=invalid_json_with_trailing_comma)
    mock_request.headers = {"content-type": "application/json"}
    mock_request.scope = {}

    # Should raise ProxyException for trailing comma
    with pytest.raises(ProxyException) as exc_info:
        await _read_request_body(mock_request)
    
    assert exc_info.value.code == "400"
    assert "Invalid JSON payload" in exc_info.value.message
    assert "trailing comma" in exc_info.value.message

    # Test case 2: Unquoted property name error
    mock_request2 = MagicMock()
    invalid_json_unquoted_property = b'''{
        "model": "gpt-4o",
        "tools": [
            {
                type: "mcp",
                "server_label": "litellm"
            }
        ],
        "input": "Run available tools"
    }'''
    
    mock_request2.body = AsyncMock(return_value=invalid_json_unquoted_property)
    mock_request2.headers = {"content-type": "application/json"}
    mock_request2.scope = {}

    # Should raise ProxyException for unquoted property
    with pytest.raises(ProxyException) as exc_info2:
        await _read_request_body(mock_request2)
    
    assert exc_info2.value.code == "400"
    assert "Invalid JSON payload" in exc_info2.value.message

    # Test case 3: Valid JSON should work normally
    mock_request3 = MagicMock()
    valid_json = b'''{
        "model": "gpt-4o",
        "tools": [
            {
                "type": "mcp",
                "server_label": "litellm",
                "headers": {
                    "x-litellm-api-key": "Bearer sk-1234"
                }
            }
        ],
        "input": "Run available tools"
    }'''
    
    mock_request3.body = AsyncMock(return_value=valid_json)
    mock_request3.headers = {"content-type": "application/json"}
    mock_request3.scope = {}

    # Should parse successfully
    result = await _read_request_body(mock_request3)
    assert result["model"] == "gpt-4o"
    assert result["input"] == "Run available tools"
    assert len(result["tools"]) == 1
    assert result["tools"][0]["type"] == "mcp"


@pytest.mark.asyncio
async def test_get_form_data():
    """
    Test that get_form_data correctly handles form data with array notation.
    Tests audio transcription parameters as a specific example.
    """
    # Create a mock request with transcription form data
    mock_request = MagicMock()

    # Create mock form data with array notation for timestamp_granularities
    mock_form_data = {
        "file": "file_object",  # In a real request this would be an UploadFile
        "model": "gpt-4o-transcribe",
        "include[]": "logprobs",  # Array notation
        "language": "en",
        "prompt": "Transcribe this audio file",
        "response_format": "json",
        "stream": "false",
        "temperature": "0.2",
        "timestamp_granularities[]": "word",  # First array item
        "timestamp_granularities[]": "segment",  # Second array item (would overwrite in dict, but handled by the function)
    }

    # Mock the form method to return the test data
    mock_request.form = AsyncMock(return_value=mock_form_data)

    # Call the function being tested
    result = await get_form_data(mock_request)

    # Verify regular form fields are preserved
    assert result["file"] == "file_object"
    assert result["model"] == "gpt-4o-transcribe"
    assert result["language"] == "en"
    assert result["prompt"] == "Transcribe this audio file"
    assert result["response_format"] == "json"
    assert result["stream"] == "false"
    assert result["temperature"] == "0.2"

    # Verify array fields are correctly parsed
    assert "include" in result
    assert isinstance(result["include"], list)
    assert "logprobs" in result["include"]

    assert "timestamp_granularities" in result
    assert isinstance(result["timestamp_granularities"], list)
    # Note: In a real MultiDict, both values would be present
    # But in our mock dictionary the second value overwrites the first
    assert "segment" in result["timestamp_granularities"]


def test_get_tags_from_request_body_with_metadata_tags():
    """
    Test that tags are correctly extracted from request body metadata.
    """
    request_body = {
        "model": "gpt-4",
        "metadata": {
            "tags": ["tag1", "tag2", "tag3"]
        }
    }
    
    result = get_tags_from_request_body(request_body=request_body)
    
    assert result == ["tag1", "tag2", "tag3"]


def test_get_tags_from_request_body_with_litellm_metadata_tags():
    """
    Test that tags are correctly extracted from request body when using litellm_metadata.
    """
    request_body = {
        "model": "gpt-4",
        "litellm_metadata": {
            "tags": ["tag1", "tag2", "tag3"]
        }
    }
    
    result = get_tags_from_request_body(request_body=request_body)
    
    assert result == ["tag1", "tag2", "tag3"]


def test_get_tags_from_request_body_with_root_tags():
    """
    Test that tags are correctly extracted from root level of request body.
    """
    request_body = {
        "model": "gpt-4",
        "tags": ["tag1", "tag2"]
    }
    
    result = get_tags_from_request_body(request_body=request_body)
    
    assert result == ["tag1", "tag2"]


def test_get_tags_from_request_body_with_combined_tags():
    """
    Test that tags from both metadata and root level are combined.
    """
    request_body = {
        "model": "gpt-4",
        "metadata": {
            "tags": ["tag1", "tag2"]
        },
        "tags": ["tag3", "tag4"]
    }
    
    result = get_tags_from_request_body(request_body=request_body)
    
    assert result == ["tag1", "tag2", "tag3", "tag4"]


def test_get_tags_from_request_body_filters_non_strings():
    """
    Test that non-string values in tags list are filtered out.
    """
    request_body = {
        "model": "gpt-4",
        "metadata": {
            "tags": ["tag1", 123, "tag2", None, "tag3", {"nested": "dict"}]
        }
    }
    
    result = get_tags_from_request_body(request_body=request_body)
    
    assert result == ["tag1", "tag2", "tag3"]


def test_get_tags_from_request_body_no_tags():
    """
    Test that empty list is returned when no tags are present.
    """
    request_body = {
        "model": "gpt-4",
        "metadata": {}
    }
    
    result = get_tags_from_request_body(request_body=request_body)
    
    assert result == []


def test_get_tags_from_request_body_with_dict_tags():
    """
    Test that function handles dict tags gracefully without crashing.
    When tags is a dict instead of a list, it should be ignored and return empty list.
    """
    request_body = {
        "model": "aws/anthropic/bedrock-claude-3-5-sonnet-v1",
        "messages": [
            {
                "role": "user",
                "content": "aloha"
            }
        ],
        "metadata": {
            "tags": {
                "litellm_id": "litellm_ratelimit_test",
                "llm_id": "llmid_ratelimit_test"
            }
        }
    }

    result = get_tags_from_request_body(request_body=request_body)

    assert result == []
    assert isinstance(result, list)


def test_get_tags_from_request_body_with_null_metadata():
    """
    Test that function handles null metadata gracefully without crashing.

    This is a regression test for https://github.com/BerriAI/litellm/issues/17263
    When metadata is explicitly set to null/None, the function should return
    an empty list instead of raising AttributeError.
    """
    request_body = {
        "model": "gpt-4",
        "metadata": None  # OpenAI API accepts metadata: null
    }

    result = get_tags_from_request_body(request_body=request_body)

    assert result == []
    assert isinstance(result, list)


def test_populate_request_with_path_params_adds_query_params():
    """
    Test that populate_request_with_path_params correctly adds query parameters
    like organization_id to the request data.
    """
    # Create a mock request with query parameters
    mock_request = MagicMock()
    # Mock query_params as a dict-like object that can be converted to dict
    mock_request.query_params = {
        "organization_id": "org-123",
        "user_id": "user-456"
    }
    mock_request.path_params = {}
    # Mock url.path to avoid errors in _add_vector_store_id_from_path
    mock_request.url.path = "/v1/chat/completions"

    # Initial request data without query params
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}]
    }

    # Call the function
    result = populate_request_with_path_params(request_data, mock_request)

    # Verify query params were added
    assert result["organization_id"] == "org-123"
    assert result["user_id"] == "user-456"
    # Verify original data is preserved
    assert result["model"] == "gpt-4"
    assert result["messages"] == [{"role": "user", "content": "Hello"}]


def test_populate_request_with_path_params_does_not_overwrite_existing_values():
    """
    Test that populate_request_with_path_params does not overwrite existing values
    in request_data when query params contain the same keys.
    """
    # Create a mock request with query parameters
    mock_request = MagicMock()
    # Mock query_params as a dict-like object that can be converted to dict
    mock_request.query_params = {
        "organization_id": "org-query-param",
        "model": "gpt-3.5-turbo"
    }
    mock_request.path_params = {}
    # Mock url.path to avoid errors in _add_vector_store_id_from_path
    mock_request.url.path = "/v1/chat/completions"

    # Initial request data with existing values
    request_data = {
        "model": "gpt-4",  # This should NOT be overwritten
        "organization_id": "org-existing",  # This should NOT be overwritten
        "messages": [{"role": "user", "content": "Hello"}]
    }

    # Call the function
    result = populate_request_with_path_params(request_data, mock_request)

    # Verify existing values were NOT overwritten
    assert result["model"] == "gpt-4"  # Should keep original, not "gpt-3.5-turbo"
    assert result["organization_id"] == "org-existing"  # Should keep original, not "org-query-param"
    # Verify other data is preserved
    assert result["messages"] == [{"role": "user", "content": "Hello"}]


@pytest.mark.asyncio
async def test_request_body_with_html_script_tags():
    """
    Test that JSON request bodies containing HTML tags like <script> are
    parsed correctly without being blocked or modified.

    Regression test for GitHub issue #20441:
    https://github.com/BerriAI/litellm/issues/20441

    LLM message content frequently contains HTML/code snippets.
    The HTTP parsing layer must not interfere with such content.
    """
    test_messages = [
        {
            "role": "user",
            "content": "<script>alert('hello')</script>",
        },
        {
            "role": "user",
            "content": "<script> test </script>",
        },
        {
            "role": "user",
            "content": "Can you explain what <script> tags do in HTML?",
        },
        {
            "role": "user",
            "content": "Here is code: <div><script src='app.js'></script></div>",
        },
        {
            "role": "user",
            "content": "<img onerror='alert(1)' src='x'>",
        },
        {
            "role": "user",
            "content": "<iframe src='https://example.com'></iframe>",
        },
    ]

    for msg in test_messages:
        test_payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello! How can I help?"},
                msg,
            ],
        }

        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=orjson.dumps(test_payload))
        mock_request.headers = {"content-type": "application/json"}
        mock_request.scope = {}

        result = await _read_request_body(mock_request)

        assert result["model"] == "gpt-4o"
        assert len(result["messages"]) == 3
        assert result["messages"][2]["content"] == msg["content"], (
            f"Message content with HTML was modified during parsing: "
            f"expected={msg['content']!r}, got={result['messages'][2]['content']!r}"
        )
