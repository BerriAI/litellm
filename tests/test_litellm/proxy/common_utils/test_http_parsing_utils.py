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
from litellm.proxy.common_utils.http_parsing_utils import (
    _read_request_body,
    _safe_get_request_parsed_body,
    _safe_set_request_parsed_body,
    get_form_data,
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
