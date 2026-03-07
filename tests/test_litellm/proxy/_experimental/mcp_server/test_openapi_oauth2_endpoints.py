"""Unit tests for openapi_oauth2_endpoints.py"""

import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, "../../../../../")

pytest.importorskip("mcp", reason="mcp package not installed; skipping MCP OAuth2 tests")

from litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints import (
    _make_state_token,
    _pending_oauth2_states,
    _purge_expired_states,
)

# ---------------------------------------------------------------------------
# _make_state_token
# ---------------------------------------------------------------------------


def test_make_state_token_returns_string():
    token = _make_state_token("server1", "user1", time.time(), "master-key")
    assert isinstance(token, str)
    assert len(token) > 0


def test_make_state_token_is_unique_for_same_inputs():
    """Two calls with identical inputs must produce different tokens (random nonce)."""
    ts = time.time()
    t1 = _make_state_token("server1", "user1", ts, "master-key")
    t2 = _make_state_token("server1", "user1", ts, "master-key")
    assert t1 != t2, "Tokens should differ due to random nonce"


def test_make_state_token_differs_by_server():
    ts = time.time()
    t1 = _make_state_token("server1", "user1", ts, "master-key")
    t2 = _make_state_token("server2", "user1", ts, "master-key")
    assert t1 != t2


# ---------------------------------------------------------------------------
# _purge_expired_states
# ---------------------------------------------------------------------------


def test_purge_expired_states_removes_expired():
    _pending_oauth2_states.clear()
    now = time.time()
    _pending_oauth2_states["expired"] = {"expires_at": now - 1}
    _pending_oauth2_states["valid"] = {"expires_at": now + 600}
    _purge_expired_states()
    assert "expired" not in _pending_oauth2_states
    assert "valid" in _pending_oauth2_states
    _pending_oauth2_states.clear()


# ---------------------------------------------------------------------------
# openapi_oauth2_connect — validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_missing_server_raises_404():
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints import (
        openapi_oauth2_connect,
    )

    mock_request = MagicMock()
    mock_user = MagicMock()
    mock_user.user_id = "user1"
    mock_user.api_key = None

    with patch(
        "litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints.global_mcp_server_manager"
    ) as mock_mgr:
        mock_mgr.get_mcp_server_by_id.return_value = None
        with pytest.raises(HTTPException) as exc_info:
            await openapi_oauth2_connect("nonexistent", mock_request, mock_user)
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_connect_missing_client_secret_raises_400():
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints import (
        openapi_oauth2_connect,
    )

    mock_request = MagicMock()
    mock_user = MagicMock()
    mock_user.user_id = "user1"
    mock_user.api_key = None

    mock_server = MagicMock()
    mock_server.authorization_url = "https://provider.example/auth"
    mock_server.token_url = "https://provider.example/token"
    mock_server.client_id = "my-client-id"
    mock_server.client_secret = None  # missing

    with patch(
        "litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints.global_mcp_server_manager"
    ) as mock_mgr, patch(
        "litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints.master_key",
        "sk-test",
        create=True,
    ):
        mock_mgr.get_mcp_server_by_id.return_value = mock_server
        with pytest.raises(HTTPException) as exc_info:
            await openapi_oauth2_connect("server1", mock_request, mock_user)
        assert exc_info.value.status_code == 400
        assert "client_secret" in exc_info.value.detail


# ---------------------------------------------------------------------------
# openapi_oauth2_callback — token parsing (provider error body)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_callback_provider_error_in_json_body():
    """HTTP 200 with JSON error body should render error page, not succeed."""
    from fastapi.responses import HTMLResponse

    from litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints import (
        openapi_oauth2_callback,
    )

    state = "test-state-123"
    now = time.time()
    _pending_oauth2_states[state] = {
        "server_id": "server1",
        "user_id": "user1",
        "timestamp": now,
        "expires_at": now + 600,
    }

    mock_server = MagicMock()
    mock_server.token_url = "https://provider.example/token"
    mock_server.client_id = "cid"
    mock_server.client_secret = "csecret"

    # Simulate provider returning HTTP 200 with an error body
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "error": "invalid_grant",
        "error_description": "Code already used",
    }
    mock_response.raise_for_status = MagicMock()  # does not raise

    mock_request = MagicMock()
    mock_request.base_url = "http://localhost:4000"

    with patch(
        "litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints.global_mcp_server_manager"
    ) as mock_mgr, patch(
        "litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints.get_request_base_url",
        return_value="http://localhost:4000",
    ), patch(
        "httpx.AsyncClient"
    ) as mock_client_cls:
        mock_mgr.get_mcp_server_by_id.return_value = mock_server
        mock_async_client = AsyncMock()
        mock_async_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await openapi_oauth2_callback(
            request=mock_request,
            code="auth-code",
            state=state,
            error=None,
            error_description=None,
        )

    assert isinstance(result, HTMLResponse)
    assert result.status_code == 502
    body_bytes = bytes(result.body) if not isinstance(result.body, bytes) else result.body
    assert b"invalid_grant" in body_bytes or b"provider" in body_bytes.lower()


# ---------------------------------------------------------------------------
# openapi_oauth2_status — basic happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_no_prisma_returns_not_connected():
    from litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints import (
        openapi_oauth2_status,
    )

    mock_user = MagicMock()
    mock_user.user_id = "user1"
    mock_user.api_key = None

    mock_server = MagicMock()
    mock_server.server_name = "GitHub"
    mock_server.name = "github"

    with patch(
        "litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints.global_mcp_server_manager"
    ) as mock_mgr, patch(
        "litellm.proxy._experimental.mcp_server.openapi_oauth2_endpoints.prisma_client",
        None,
        create=True,
    ):
        mock_mgr.get_mcp_server_by_id.return_value = mock_server
        result = await openapi_oauth2_status("server1", mock_user)

    import json

    raw = result.body
    body = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw))
    assert body["connected"] is False
    assert body["server_id"] == "server1"
