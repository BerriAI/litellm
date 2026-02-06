import os
import sys
import unittest.mock as mock

import pytest
from httpx import Response

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm_enterprise.enterprise_callbacks.send_emails.resend_email import (
    ResendEmailLogger,
)

# Test file for Resend email integration


@pytest.fixture(autouse=True)
def clear_client_cache():
    """
    Clear the HTTP client cache before each test to ensure mocks are used.
    This prevents cached real clients from being reused across tests.
    """
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is not None:
        cache.flush_cache()
    yield
    # Clear again after test to avoid polluting other tests
    if cache is not None:
        cache.flush_cache()


@pytest.fixture
def mock_env_vars():
    with mock.patch.dict(os.environ, {"RESEND_API_KEY": "test_api_key"}):
        yield


@pytest.mark.asyncio
async def test_send_email_success(mock_env_vars):
    # Initialize the logger
    logger = ResendEmailLogger()

    # Test data
    from_email = "test@example.com"
    to_email = ["recipient@example.com"]
    subject = "Test Subject"
    html_body = "<p>Test email body</p>"

    # Create mock HTTP client and inject it directly into the logger
    # This ensures the mock is used regardless of any caching/import issues
    mock_response = mock.Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "test_email_id"}
    mock_response.raise_for_status.return_value = None

    mock_async_client = mock.AsyncMock()
    mock_async_client.post.return_value = mock_response

    # Directly inject the mock client to bypass any caching
    logger.async_httpx_client = mock_async_client

    # Send email
    await logger.send_email(
        from_email=from_email, to_email=to_email, subject=subject, html_body=html_body
    )

    # Verify the HTTP client was called correctly
    mock_async_client.post.assert_called_once()
    call_args = mock_async_client.post.call_args

    # Verify the URL
    assert call_args[1]["url"] == "https://api.resend.com/emails"

    # Verify the request body
    request_body = call_args[1]["json"]
    assert request_body["from"] == from_email
    assert request_body["to"] == to_email
    assert request_body["subject"] == subject
    assert request_body["html"] == html_body

    # Verify the headers
    assert call_args[1]["headers"] == {"Authorization": "Bearer test_api_key"}


@pytest.mark.asyncio
async def test_send_email_missing_api_key():
    # Remove the API key from environment before initializing logger
    original_key = os.environ.pop("RESEND_API_KEY", None)
    
    try:
        # Initialize the logger after removing the API key
        logger = ResendEmailLogger()

        # Test data
        from_email = "test@example.com"
        to_email = ["recipient@example.com"]
        subject = "Test Subject"
        html_body = "<p>Test email body</p>"

        # Create mock HTTP client and inject it directly into the logger
        # This ensures the mock is used regardless of any caching issues
        mock_response = mock.Mock(spec=Response)
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "test_email_id"}
        
        mock_async_client = mock.AsyncMock()
        mock_async_client.post.return_value = mock_response
        
        # Directly inject the mock client to bypass any caching
        logger.async_httpx_client = mock_async_client

        # Send email
        await logger.send_email(
            from_email=from_email, to_email=to_email, subject=subject, html_body=html_body
        )

        # Verify the HTTP client was called with None as the API key
        mock_async_client.post.assert_called_once()
        call_args = mock_async_client.post.call_args
        assert call_args[1]["headers"] == {"Authorization": "Bearer None"}
    finally:
        # Restore the original key if it existed
        if original_key is not None:
            os.environ["RESEND_API_KEY"] = original_key


@pytest.mark.asyncio
async def test_send_email_multiple_recipients(mock_env_vars):
    # Initialize the logger
    logger = ResendEmailLogger()

    # Test data with multiple recipients
    from_email = "test@example.com"
    to_email = ["recipient1@example.com", "recipient2@example.com"]
    subject = "Test Subject"
    html_body = "<p>Test email body</p>"

    # Create mock HTTP client and inject it directly into the logger
    mock_response = mock.Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "test_email_id"}
    mock_response.raise_for_status.return_value = None

    mock_async_client = mock.AsyncMock()
    mock_async_client.post.return_value = mock_response

    # Directly inject the mock client to bypass any caching
    logger.async_httpx_client = mock_async_client

    # Send email
    await logger.send_email(
        from_email=from_email, to_email=to_email, subject=subject, html_body=html_body
    )

    # Verify the HTTP client was called with multiple recipients
    mock_async_client.post.assert_called_once()
    call_args = mock_async_client.post.call_args
    request_body = call_args[1]["json"]
    assert request_body["to"] == to_email
