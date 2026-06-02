import smtplib
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy.utils import _get_openapi_url, send_email


@pytest.mark.parametrize(
    "env_vars, expected_url",
    [
        ({}, "/openapi.json"),  # default case
        ({"NO_OPENAPI": "True"}, None),  # OpenAPI disabled
    ],
)
def test_get_openapi_url(monkeypatch, env_vars, expected_url):
    # Clear relevant environment variables
    monkeypatch.delenv("NO_OPENAPI", raising=False)

    # Set test environment variables
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    result = _get_openapi_url()
    assert result == expected_url


# ── SMTP connection selection tests ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_email_port_465_uses_smtp_ssl(monkeypatch):
    """Port 465 must use SMTP_SSL (implicit SSL), not SMTP + starttls."""
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "465")
    monkeypatch.setenv("SMTP_SENDER_EMAIL", "noreply@example.com")
    monkeypatch.delenv("SMTP_USE_SSL", raising=False)

    mock_server = MagicMock()
    mock_server.__enter__ = lambda s: s
    mock_server.__exit__ = MagicMock(return_value=False)

    with (
        patch("smtplib.SMTP_SSL", return_value=mock_server) as mock_ssl,
        patch("smtplib.SMTP") as mock_plain,
    ):
        await send_email(
            receiver_email="user@example.com",
            subject="Test",
            html="<p>Hi</p>",
        )
        mock_ssl.assert_called_once_with(host="smtp.example.com", port=465)
        mock_plain.assert_not_called()
        mock_server.starttls.assert_not_called()


@pytest.mark.asyncio
async def test_send_email_smtp_use_ssl_env_forces_ssl(monkeypatch):
    """SMTP_USE_SSL=True must use SMTP_SSL regardless of port."""
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "2525")
    monkeypatch.setenv("SMTP_USE_SSL", "True")
    monkeypatch.setenv("SMTP_SENDER_EMAIL", "noreply@example.com")

    mock_server = MagicMock()
    mock_server.__enter__ = lambda s: s
    mock_server.__exit__ = MagicMock(return_value=False)

    with (
        patch("smtplib.SMTP_SSL", return_value=mock_server) as mock_ssl,
        patch("smtplib.SMTP") as mock_plain,
    ):
        await send_email(
            receiver_email="user@example.com",
            subject="Test",
            html="<p>Hi</p>",
        )
        mock_ssl.assert_called_once_with(host="smtp.example.com", port=2525)
        mock_plain.assert_not_called()
        mock_server.starttls.assert_not_called()


@pytest.mark.asyncio
async def test_send_email_smtp_use_ssl_env_lowercase(monkeypatch):
    """SMTP_USE_SSL=true (lowercase) must also trigger SMTP_SSL."""
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "2525")
    monkeypatch.setenv("SMTP_USE_SSL", "true")
    monkeypatch.setenv("SMTP_SENDER_EMAIL", "noreply@example.com")

    mock_server = MagicMock()
    mock_server.__enter__ = lambda s: s
    mock_server.__exit__ = MagicMock(return_value=False)

    with (
        patch("smtplib.SMTP_SSL", return_value=mock_server) as mock_ssl,
        patch("smtplib.SMTP") as mock_plain,
    ):
        await send_email(
            receiver_email="user@example.com",
            subject="Test",
            html="<p>Hi</p>",
        )
        mock_ssl.assert_called_once_with(host="smtp.example.com", port=2525)
        mock_plain.assert_not_called()


@pytest.mark.asyncio
async def test_send_email_port_587_uses_starttls(monkeypatch):
    """Port 587 (default) must use SMTP + starttls — backwards compat regression guard."""
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_SENDER_EMAIL", "noreply@example.com")
    monkeypatch.delenv("SMTP_USE_SSL", raising=False)
    monkeypatch.delenv("SMTP_TLS", raising=False)

    mock_server = MagicMock()
    mock_server.__enter__ = lambda s: s
    mock_server.__exit__ = MagicMock(return_value=False)

    with (
        patch("smtplib.SMTP", return_value=mock_server) as mock_plain,
        patch("smtplib.SMTP_SSL") as mock_ssl,
    ):
        await send_email(
            receiver_email="user@example.com",
            subject="Test",
            html="<p>Hi</p>",
        )
        mock_plain.assert_called_once_with(host="smtp.example.com", port=587)
        mock_ssl.assert_not_called()
        mock_server.starttls.assert_called_once()
