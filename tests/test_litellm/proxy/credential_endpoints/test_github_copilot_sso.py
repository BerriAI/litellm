"""
Tests for the GitHub Copilot SSO (device code) credential endpoints.

The backend is fully stateless — no DB writes during the auth flow.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ASYNC_HTTP_PATCH = "litellm.proxy.credential_endpoints.github_copilot_sso.get_async_httpx_client"


def _make_async_client(json_data):
    """Build a mock client whose .post() returns a single response."""
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=json_data)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    return mock_client


# ---------------------------------------------------------------------------
# Tests for initiate endpoint
# ---------------------------------------------------------------------------


class TestGithubCopilotInitiate:
    @pytest.mark.asyncio
    async def test_initiate_success(self):
        """POST /initiate returns device_code, user_code, verification_uri, poll_interval_ms, expires_in."""
        ctx = _make_async_client({
            "device_code": "test-device-code-123",
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.com/login/device",
            "interval": 5,
            "expires_in": 900,
        })

        with patch(ASYNC_HTTP_PATCH, return_value=ctx):
            from litellm.proxy.credential_endpoints.github_copilot_sso import (
                github_copilot_initiate,
            )

            result = await github_copilot_initiate(MagicMock(), MagicMock(), MagicMock())
            assert result.device_code == "test-device-code-123"
            assert result.user_code == "ABCD-1234"
            assert result.verification_uri == "https://github.com/login/device"
            assert result.poll_interval_ms == 5000
            assert result.expires_in == 900

    @pytest.mark.asyncio
    async def test_initiate_github_error(self):
        """POST /initiate returns 502 when GitHub device code API fails."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("network error"))

        with patch(ASYNC_HTTP_PATCH, return_value=mock_client):
            from fastapi import HTTPException

            from litellm.proxy.credential_endpoints.github_copilot_sso import (
                github_copilot_initiate,
            )

            with pytest.raises(HTTPException) as exc_info:
                await github_copilot_initiate(MagicMock(), MagicMock(), MagicMock())
            assert exc_info.value.status_code == 502


# ---------------------------------------------------------------------------
# Tests for status endpoint
# ---------------------------------------------------------------------------


class TestGithubCopilotStatus:
    @pytest.mark.asyncio
    async def test_status_pending(self):
        """POST /status returns pending when GitHub says authorization_pending."""
        ctx = _make_async_client({"error": "authorization_pending"})

        with patch(ASYNC_HTTP_PATCH, return_value=ctx):
            from litellm.proxy.credential_endpoints.github_copilot_sso import (
                StatusRequest,
                github_copilot_status,
            )

            result = await github_copilot_status(
                MagicMock(), MagicMock(), StatusRequest(device_code="test-dc"), MagicMock()
            )
            assert result.status == "pending"
            assert result.access_token is None
            assert result.retry_after_ms is None

    @pytest.mark.asyncio
    async def test_status_slow_down(self):
        """POST /status returns pending with retry_after_ms when GitHub says slow_down."""
        ctx = _make_async_client({"error": "slow_down", "interval": 10})

        with patch(ASYNC_HTTP_PATCH, return_value=ctx):
            from litellm.proxy.credential_endpoints.github_copilot_sso import (
                StatusRequest,
                github_copilot_status,
            )

            result = await github_copilot_status(
                MagicMock(), MagicMock(), StatusRequest(device_code="test-dc"), MagicMock()
            )
            assert result.status == "pending"
            assert result.retry_after_ms == 10_000

    @pytest.mark.asyncio
    async def test_status_complete(self):
        """POST /status returns access_token on success."""
        ctx = _make_async_client({"access_token": "ghu_abc123"})

        with patch(ASYNC_HTTP_PATCH, return_value=ctx):
            from litellm.proxy.credential_endpoints.github_copilot_sso import (
                StatusRequest,
                github_copilot_status,
            )

            result = await github_copilot_status(
                MagicMock(), MagicMock(), StatusRequest(device_code="test-dc"), MagicMock()
            )
            assert result.status == "complete"
            assert result.access_token == "ghu_abc123"
            assert result.error is None

    @pytest.mark.asyncio
    async def test_status_slow_down_missing_interval(self):
        """POST /status returns failed when GitHub slow_down has no interval field."""
        ctx = _make_async_client({"error": "slow_down"})

        with patch(ASYNC_HTTP_PATCH, return_value=ctx):
            from litellm.proxy.credential_endpoints.github_copilot_sso import (
                StatusRequest,
                github_copilot_status,
            )

            result = await github_copilot_status(
                MagicMock(), MagicMock(), StatusRequest(device_code="test-dc"), MagicMock()
            )
            assert result.status == "failed"
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_status_failed(self):
        """POST /status returns failed when GitHub returns expired_token."""
        ctx = _make_async_client({
            "error": "expired_token",
            "error_description": "The device code has expired.",
        })

        with patch(ASYNC_HTTP_PATCH, return_value=ctx):
            from litellm.proxy.credential_endpoints.github_copilot_sso import (
                StatusRequest,
                github_copilot_status,
            )

            result = await github_copilot_status(
                MagicMock(), MagicMock(), StatusRequest(device_code="test-dc"), MagicMock()
            )
            assert result.status == "failed"
            assert "expired" in (result.error or "").lower()
