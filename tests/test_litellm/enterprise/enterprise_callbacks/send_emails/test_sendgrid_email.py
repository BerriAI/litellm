import os
import sys
import unittest.mock as mock

import pytest
from httpx import Response

sys.path.insert(0, os.path.abspath("../../.."))

from litellm_enterprise.enterprise_callbacks.send_emails.sendgrid_email import (
    SendGridEmailLogger,
)


@pytest.fixture
def mock_env_vars():
    with mock.patch.dict(os.environ, {"SENDGRID_API_KEY": "test_api_key"}):
        yield


@pytest.fixture
def mock_httpx_client():
    with mock.patch(
        "litellm_enterprise.enterprise_callbacks.send_emails.sendgrid_email.get_async_httpx_client"
    ) as mock_client:
        mock_response = mock.AsyncMock(spec=Response)
        mock_response.status_code = 202
        mock_response.text = "accepted"

        mock_async_client = mock.AsyncMock()
        mock_async_client.post.return_value = mock_response
        mock_client.return_value = mock_async_client

        yield mock_async_client


@pytest.mark.asyncio
async def test_send_email_success(mock_env_vars, mock_httpx_client):
    logger = SendGridEmailLogger()

    from_email = "test@example.com"
    to_email = ["recipient@example.com"]
    subject = "Test Subject"
    html_body = "<p>Test email body</p>"

    await logger.send_email(
        from_email=from_email, to_email=to_email, subject=subject, html_body=html_body
    )

    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args
    assert call_args[1]["url"] == "https://api.sendgrid.com/v3/mail/send"

    payload = call_args[1]["json"]
    assert payload["from"] == {"email": from_email}
    assert payload["personalizations"][0]["to"] == [{"email": to_email[0]}]
    assert payload["personalizations"][0]["subject"] == subject
    assert payload["content"][0]["type"] == "text/html"
    assert payload["content"][0]["value"] == html_body

    assert call_args[1]["headers"] == {"Authorization": "Bearer test_api_key"}


@pytest.mark.asyncio
async def test_send_email_missing_api_key(mock_httpx_client):
    with mock.patch.dict(os.environ, {}, clear=True):
        logger = SendGridEmailLogger()

        with pytest.raises(ValueError):
            await logger.send_email(
                from_email="test@example.com",
                to_email=["recipient@example.com"],
                subject="Test Subject",
                html_body="<p>Test email body</p>",
            )

        mock_httpx_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_send_email_multiple_recipients(mock_env_vars, mock_httpx_client):
    logger = SendGridEmailLogger()

    from_email = "test@example.com"
    to_email = ["recipient1@example.com", "recipient2@example.com"]
    subject = "Test Subject"
    html_body = "<p>Test email body</p>"

    await logger.send_email(
        from_email=from_email, to_email=to_email, subject=subject, html_body=html_body
    )

    mock_httpx_client.post.assert_called_once()
    payload = mock_httpx_client.post.call_args[1]["json"]

    assert payload["personalizations"][0]["to"] == [
        {"email": "recipient1@example.com"},
        {"email": "recipient2@example.com"},
    ]
