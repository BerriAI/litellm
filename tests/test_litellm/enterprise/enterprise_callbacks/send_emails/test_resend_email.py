import os
import sys
import unittest.mock as mock

import pytest
from httpx import Response

sys.path.insert(0, os.path.abspath("../../.."))

from litellm_enterprise.enterprise_callbacks.send_emails.resend_email import (
    ResendEmailLogger,
)


@pytest.fixture
def mock_env_vars():
    with mock.patch.dict(os.environ, {"RESEND_API_KEY": "test_api_key"}):
        yield


@pytest.fixture
def mock_httpx_client():
    with mock.patch(
        "litellm_enterprise.enterprise_callbacks.send_emails.resend_email.get_async_httpx_client"
    ) as mock_client:
        # Create a mock response
        mock_response = mock.AsyncMock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "test_email_id"}

        # Create a mock client
        mock_async_client = mock.AsyncMock()
        mock_async_client.post.return_value = mock_response
        mock_client.return_value = mock_async_client

        yield mock_async_client


@pytest.mark.asyncio
async def test_send_email_success(mock_env_vars, mock_httpx_client):
    # Initialize the logger
    logger = ResendEmailLogger()

    # Test data
    from_email = "test@example.com"
    to_email = ["recipient@example.com"]
    subject = "Test Subject"
    html_body = "<p>Test email body</p>"

    # Send email
    await logger.send_email(
        from_email=from_email, to_email=to_email, subject=subject, html_body=html_body
    )

    # Verify the HTTP client was called correctly
    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args

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
async def test_send_email_missing_api_key(mock_httpx_client):
    # Remove the API key from environment
    if "RESEND_API_KEY" in os.environ:
        del os.environ["RESEND_API_KEY"]

    # Initialize the logger
    logger = ResendEmailLogger()

    # Test data
    from_email = "test@example.com"
    to_email = ["recipient@example.com"]
    subject = "Test Subject"
    html_body = "<p>Test email body</p>"

    # Send email
    await logger.send_email(
        from_email=from_email, to_email=to_email, subject=subject, html_body=html_body
    )

    # Verify the HTTP client was called with None as the API key
    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args
    assert call_args[1]["headers"] == {"Authorization": "Bearer None"}


@pytest.mark.asyncio
async def test_send_email_multiple_recipients(mock_env_vars, mock_httpx_client):
    # Initialize the logger
    logger = ResendEmailLogger()

    # Test data with multiple recipients
    from_email = "test@example.com"
    to_email = ["recipient1@example.com", "recipient2@example.com"]
    subject = "Test Subject"
    html_body = "<p>Test email body</p>"

    # Send email
    await logger.send_email(
        from_email=from_email, to_email=to_email, subject=subject, html_body=html_body
    )

    # Verify the HTTP client was called with multiple recipients
    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args
    request_body = call_args[1]["json"]
    assert request_body["to"] == to_email
