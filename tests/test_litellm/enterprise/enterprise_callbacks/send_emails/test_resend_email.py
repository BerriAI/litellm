import os
import sys
import unittest.mock as mock

import httpx
import pytest
import respx
from httpx import Response

sys.path.insert(0, os.path.abspath("../../.."))

from litellm_enterprise.enterprise_callbacks.send_emails.resend_email import (
    ResendEmailLogger,
)

# Test file for Resend email integration


@pytest.fixture
def mock_env_vars():
    with mock.patch.dict(os.environ, {"RESEND_API_KEY": "test_api_key"}):
        yield


@pytest.fixture
def mock_httpx_client():
    with mock.patch(
        "litellm_enterprise.enterprise_callbacks.send_emails.resend_email.get_async_httpx_client"
    ) as mock_client:

        mock_response = mock.Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "test_email_id"}
        mock_response.raise_for_status.return_value = None 

        mock_async_client = mock.AsyncMock()
        mock_async_client.post.return_value = mock_response

        mock_client.return_value = mock_async_client
        yield mock_async_client


@pytest.mark.asyncio
@respx.mock
async def test_send_email_success(mock_env_vars, mock_httpx_client):
    # Block all HTTP requests at network level to prevent real API calls
    respx.post("https://api.resend.com/emails").mock(
        return_value=httpx.Response(200, json={"id": "test_email_id"})
    )
    
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
@respx.mock
async def test_send_email_missing_api_key(mock_httpx_client):
    # Block all HTTP requests at network level to prevent real API calls
    respx.post("https://api.resend.com/emails").mock(
        return_value=httpx.Response(200, json={"id": "test_email_id"})
    )
    
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

        # Mock the response to avoid making real HTTP requests
        mock_response = mock.Mock(spec=Response)
        mock_response.raise_for_status.return_value = None

        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "test_email_id"}
        mock_httpx_client.post.return_value = mock_response

        # Send email
        await logger.send_email(
            from_email=from_email, to_email=to_email, subject=subject, html_body=html_body
        )

        # Verify the HTTP client was called with None as the API key
        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        assert call_args[1]["headers"] == {"Authorization": "Bearer None"}
    finally:
        # Restore the original key if it existed
        if original_key is not None:
            os.environ["RESEND_API_KEY"] = original_key


@pytest.mark.asyncio
@respx.mock
async def test_send_email_multiple_recipients(mock_env_vars, mock_httpx_client):
    # Block all HTTP requests at network level to prevent real API calls
    respx.post("https://api.resend.com/emails").mock(
        return_value=httpx.Response(200, json={"id": "test_email_id"})
    )
    
    # Initialize the logger
    logger = ResendEmailLogger()

    # Test data with multiple recipients
    from_email = "test@example.com"
    to_email = ["recipient1@example.com", "recipient2@example.com"]
    subject = "Test Subject"
    html_body = "<p>Test email body</p>"

    # Mock the response to avoid making real HTTP requests
    mock_response = mock.Mock(spec=Response)
    mock_response.raise_for_status.return_value = None

    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "test_email_id"}
    mock_httpx_client.post.return_value = mock_response

    # Send email
    await logger.send_email(
        from_email=from_email, to_email=to_email, subject=subject, html_body=html_body
    )

    # Verify the HTTP client was called with multiple recipients
    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args
    request_body = call_args[1]["json"]
    assert request_body["to"] == to_email
