import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from starlette.datastructures import Headers
from starlette.requests import HTTPConnection
import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy._types import ProxyException


@pytest.mark.asyncio
async def test_read_request_body_valid_json():
    """Test the function with a valid JSON payload."""

    class MockRequest:
        async def body(self):
            return b'{"key": "value"}'

    request = MockRequest()
    result = await _read_request_body(request)
    assert result == {"key": "value"}


@pytest.mark.asyncio
async def test_read_request_body_empty_body():
    """Test the function with an empty body."""

    class MockRequest:
        async def body(self):
            return b""

    request = MockRequest()
    result = await _read_request_body(request)
    assert result == {}


@pytest.mark.asyncio
async def test_read_request_body_invalid_json():
    """Test the function with an invalid JSON payload."""

    class MockRequest:
        async def body(self):
            return b'{"key": value}'  # Missing quotes around `value`

    request = MockRequest()
    with pytest.raises(ProxyException):
        await _read_request_body(request)


@pytest.mark.asyncio
async def test_read_request_body_large_payload():
    """Test the function with a very large payload."""
    large_payload = '{"key":' + '"a"' * 10**6 + "}"  # Large payload

    class MockRequest:
        async def body(self):
            return large_payload.encode()

    request = MockRequest()
    with pytest.raises(ProxyException):
        await _read_request_body(request)


@pytest.mark.asyncio
async def test_read_request_body_unexpected_error():
    """Test the function when an unexpected error occurs."""

    class MockRequest:
        async def body(self):
            raise ValueError("Unexpected error")

    request = MockRequest()
    result = await _read_request_body(request)
    assert result == {}  # Ensure fallback behavior
