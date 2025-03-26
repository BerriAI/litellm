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
    assert mock_request.scope["parsed_body"] == test_data

    # Verify the body was read once
    mock_request.body.assert_called_once()

    # Reset the mock to track the second call
    mock_request.body.reset_mock()

    # Second call should use the cached body
    result2 = await _read_request_body(mock_request)
    assert result2 == test_data

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
    assert mock_request.scope["parsed_body"] == test_data

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
    assert mock_request.scope["parsed_body"] == {}

    # Verify the body was read
    mock_request.body.assert_called_once()
